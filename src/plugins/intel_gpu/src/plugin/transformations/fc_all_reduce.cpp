// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_all_reduce.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/add.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/op/sync_tensor.hpp"
#include "intel_gpu/op/util.hpp"
#include "openvino/op/slice.hpp"
#include <cstdlib>

namespace ov {
namespace intel_gpu {

FullyConnectedSplitInput::FullyConnectedSplitInput(size_t world_size, size_t rank_size) {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto weights = any_input();
    auto bias = any_input();
    auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias}, consumers_count(1));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_bias = pattern_map.at(bias).get_node_shared_ptr();
        const auto& m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();

        std::map<int, std::shared_ptr<ov::Node>> org_users;
        for (auto u : m_fc->get_users()) {
            for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                if (u->get_input_node_shared_ptr(idx) == m_fc) {
                    org_users.insert({idx, u});
                }
            }
        }

        int w_rank = rank_size;
        int w_size = world_size;
        if (w_size != 1) {
            int slice_axis_length = m_data->get_output_partial_shape(0)[-1].get_length();
            auto scop = std::div(slice_axis_length, w_size).quot;

            auto start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {w_rank * scop});
            auto stop = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {(w_rank + 1) * scop});
            auto step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});

            int64_t input_axis_value = m_data->get_output_partial_shape(0).size() - 1;
            auto input_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {input_axis_value});
            auto data_slice = std::make_shared<ov::op::v8::Slice>(m_data, start, stop, step, input_axis);

            int64_t weights_axis_value = m_weights->get_shape().size() - 1;
            auto weights_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {weights_axis_value});
            auto weights_slice = std::make_shared<ov::op::v8::Slice>(m_weights, start, stop, step, weights_axis);

            auto fc_0 =
                std::make_shared<op::FullyConnected>(data_slice, weights_slice, m_bias, m_fc->get_element_type());

            std::shared_ptr<ov::intel_gpu::op::SyncTensor> sync_node;
            sync_node = std::make_shared<ov::intel_gpu::op::SyncTensor>(fc_0,
                                                                        w_size,
                                                                        m_weights->get_shape()[-1],
                                                                        m_fc->get_element_type(),
                                                                        ov::intel_gpu::op::TP_MODE::ALL_REDUCE);
            sync_node->set_friendly_name(m_fc->get_friendly_name() + "_TP");
            copy_runtime_info(m_fc, fc_0);
            for (auto& iter : org_users) {
                iter.second->input(iter.first).replace_source_output(sync_node->output(0));
            }
            m_fc->clear_control_dependencies();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected, "FullyConnectedSplitInput");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
