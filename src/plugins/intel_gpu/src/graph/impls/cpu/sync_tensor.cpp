// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_tensor_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"

namespace cldnn {
namespace cpu {

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<sync_tensor_impl>(*this);
    }

    sync_tensor_impl() : parent() {}

    explicit sync_tensor_impl(const sync_tensor_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<sync_tensor>());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();
        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        std::cout << "start!" << "id = "<< id << std::endl;
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }
        for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
            if (idx != w_rank) {
                sub_mem_mgr->_memorys_table[id][w_rank].recv_buf[idx].mem = instance.output_memory(idx).buffer_ptr();
                sub_mem_mgr->_memorys_table[id][w_rank].recv_buf[idx].ready_flag = false;
            }
        }
        std::vector<int> copy_list(w_size, 1);
        copy_list[w_rank] = 0;
        // write buffer to peers
        while (true) {
            int copy_size = 0;
            std::cout << "rank: " << w_rank << "trying to sent buffer to rank: " << std::endl;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && copy_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].recv_buf[w_rank].mem) {
                    auto src_ptr = static_cast<uint8_t*>(instance.output_memory(w_rank).buffer_ptr());
                    auto dst_ptr = sub_mem_mgr->_memorys_table[id][idx].recv_buf[w_rank].mem;
                    std::memcpy(dst_ptr, src_ptr, instance.output_memory(w_rank).size());
                    copy_list[idx] = 0;
                    sub_mem_mgr->_memorys_table[id][idx].recv_buf[w_rank].ready_flag = true;
                    std::cout << "rank: " << w_rank << ", sent buffer to rank: " << idx << std::endl;
                }
                copy_size += copy_list[idx];
            }
            if (copy_size == 0) {
                std::cout << "rank: " << w_rank << ", sending buffer done!" << std::endl;
                break;
            }
        }
        std::vector<int> wait_list(w_size, 1);
        wait_list[w_rank] = 0; // no need to wait for itself
        /*while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && sub_mem_mgr->_memorys_table[id][w_rank].recv_buf[idx].ready_flag) {
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                std::cout << "rank:" << w_rank << ", sync tensor done!" << std::endl;
                break;
            }
        }*/
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_use_count[id]++;
            std::cout << "id: " << id << ", use_count: " << sub_mem_mgr->_use_count[id] << std::endl;
        }
        std::cout << "finished!" << std::endl;


        std::cout << "******************************" << std::endl;
        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }
        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<sync_tensor_impl>();
    }
};


namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::cpu, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::cpu, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)