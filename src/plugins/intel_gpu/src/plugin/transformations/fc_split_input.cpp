// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_split_input.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "intel_gpu/graph/ccl_messenger.hpp"
#include <cstdlib>

namespace ov {
namespace intel_gpu {

FullyConnectedSplitInput::FullyConnectedSplitInput() {
    using namespace ov::pass::pattern;

    auto data = any_input();
    auto weights = any_input();
    auto bias = any_input();
    auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias}, consumers_count(1));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        std::cout << "run FullyConnectedSplitInput ......................" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& m_data = pattern_map.at(data).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_bias = pattern_map.at(bias).get_node_shared_ptr();
        const auto& m_fc = pattern_map.at(fully_connected).get_node_shared_ptr();

        int w_rank{0}, w_size{1};
        if (getenv("ENABLE_CCL")) {  // only apply for dynamic models for now
            w_rank = cldnn::Messenger::getInstance().getRank();
            w_size = cldnn::Messenger::getInstance().getSize();
        }
        // std::cout << "Apply TP rank " << w_rank << ", size: " << w_size << std::endl;
        if (w_size != 1) {
            if (m_data->get_output_partial_shape(0).is_dynamic())
                std::cout << "m_data shape: " << m_data->get_output_partial_shape(0) << std::endl;
            int64_t input_axis_value = m_data->get_shape().size() - 1;
            const auto input_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                           ov::Shape{1},
                                                                           std::vector<int64_t>{input_axis_value});
            std::vector<int64_t> input_parts(w_size, std::div(m_data->get_shape()[input_axis_value], w_size).quot);
            input_parts[w_size - 1] = -1;
            const auto input_splitLengths =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{(unsigned long)w_size}, input_parts);
            auto input_split = std::make_shared<ov::op::v1::VariadicSplit>(m_data, input_axis, input_splitLengths);

            int64_t weight_axis_value = m_weights->get_shape().size() - 1;
            const auto weight_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                            ov::Shape{1},
                                                                            std::vector<int64_t>{weight_axis_value});
            std::vector<int64_t> weight_parts(w_size, std::div(m_weights->get_shape()[weight_axis_value], w_size).quot);
            weight_parts[w_size - 1] = -1;
            const auto weight_splitLengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                                    ov::Shape{(unsigned long)w_size},
                                                                                    weight_parts);
            auto weight_split =
                std::make_shared<ov::op::v1::VariadicSplit>(m_weights, weight_axis, weight_splitLengths);

            auto fc_0 = std::make_shared<op::FullyConnected>(input_split->output(w_rank),
                                                             weight_split->output(w_rank),
                                                             m_bias,
                                                             ov::element::f32);
            replace_node(m_fc, fc_0);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected, "FullyConnectedSplitInput");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
