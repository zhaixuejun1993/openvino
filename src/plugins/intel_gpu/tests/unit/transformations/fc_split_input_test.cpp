// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include "openvino/openvino.hpp"
#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include <plugin/transformations/fc_split_input.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/graph/ccl_messenger.hpp"

#include <chrono>
using namespace std::chrono;

using namespace testing;
using namespace ov::intel_gpu;

TEST(TransformationTestsF1, FullyConnectedSplitInput1) {
    // {
    //     auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
    //     auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
    //     auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    //     auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);

    //     model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    // }

    // {
    //     auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5, 2, 2});
    //     const auto axis =
    //         std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    //     const auto splitLengths =
    //         std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{3, 2});
    //     auto Node = std::make_shared<ov::op::v1::VariadicSplit>(input1, axis, splitLengths);
    //     auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
    //     auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    //     auto matmul = std::make_shared<op::FullyConnected>(Node->output(0), input2, no_bias, ov::element::f32);
    //     model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    //     manager.register_pass<FullyConnectedSplitInput>();
    //     ov::serialize(model, "./model_split1.xml", "./model_split1.bin");
    // }

    {
        // -------- Construct model
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1, 2, 3, 4});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        // auto no_bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {0});
        // auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);

        int64_t input_axis_value{1};
        const auto input_axis =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{input_axis_value});
        const auto input_splitLengths =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, -1});
        auto input_split = std::make_shared<ov::op::v1::VariadicSplit>(input1, input_axis, input_splitLengths);

        int64_t weight_axis_value{1};
        const auto weight_axis =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{weight_axis_value});
        const auto weight_splitLengths =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, -1});
        auto weight_split = std::make_shared<ov::op::v1::VariadicSplit>(input2, weight_axis, weight_splitLengths);

        auto fc_0 = std::make_shared<op::FullyConnected>(input_split->output(0), weight_split->output(0), no_bias, ov::element::f32);
        auto fc_1 = std::make_shared<op::FullyConnected>(input_split->output(1), weight_split->output(1), no_bias, ov::element::f32);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{fc_0, fc_1}, ov::ParameterVector{input1});
        ov::serialize(model, "./model_fc.xml", "./model_fc.bin");

        // ov::pass::Manager manager;
        // manager.register_pass<FullyConnectedSplitInput>();
        // manager.run_passes(model);
        // ov::serialize(model, "./model_fc_pass.xml", "./model_fc_pass.bin");

        // -------- Loading a model to the device --------
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
        ov::serialize(compiled_model.get_runtime_model(), "./model_fc_compile.xml", "./model_fc_compile.bin");
        

        // -------- Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Prepare input --------
        auto input_generate = ov::test::utils::InputGenerateData(0, 5);
        auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(), infer_request.get_input_tensor().get_shape(), input_generate);
        std::cout << "\n" << "input_tensor: ";
        for (size_t i = 0; i < tensor.get_size(); i++) {
            std::cout << tensor.data<float>()[i] << ", ";
        }
        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();

        // -------- Process output
        // const ov::Tensor& output_tensor = infer_request.get_output_tensor();
        auto output_tensor_fc_0 = infer_request.get_output_tensor(0);
        std::cout << "\n" << "output_tensor_fc_0: ";
        for (size_t i = 0; i < output_tensor_fc_0.get_size(); i++) {
            std::cout << output_tensor_fc_0.data<float>()[i] << ", ";
        }

        auto output_tensor_fc_1 = infer_request.get_output_tensor(1);
        std::cout << "\n" << "output_tensor_fc_1: ";
        for (size_t i = 0; i < output_tensor_fc_1.get_size(); i++) {
            std::cout << output_tensor_fc_1.data<float>()[i] << ", ";
        }

        // std::cout << "\n" << "output_tensor: ";
        // for (size_t i = 0; i < output_tensor.get_size(); i++) {
        //     std::cout << output_tensor.data<float>()[i] << ", ";
        // }

        // ov::pass::Manager manager;
        // manager.register_pass<FullyConnectedSplitInput>();
        // manager.run_passes(model);
        // ov::serialize(model, "./model_fc_pass.xml", "./model_fc_pass.bin");
    
    }

    // {
    //     ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 8, 2, 2}})};
    //     const auto axis =
    //         std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    //     const auto splitLengths =
    //         std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 2, -1});
    //     auto Node = std::make_shared<ov::op::v1::VariadicSplit>(params.at(0), axis, splitLengths);
    //     ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node->output(0)),
    //                              std::make_shared<ov::op::v0::Result>(Node->output(1)),
    //                              std::make_shared<ov::op::v0::Result>(Node->output(2)),
    //                              std::make_shared<ov::op::v0::Result>(Node->output(3))};
    //     auto model_sp = std::make_shared<ov::Model>(results, params, "VariadicSplitGraph");
    //     ov::serialize(model_sp, "./model_sp.xml", "./model_sp.bin");
    // }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput2) {

    {
        // -------- Construct model
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1, 2, 3, 4});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
        ov::serialize(model, "./model_fc.xml", "./model_fc.bin");

        // -------- Loading a model to the device --------
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
        ov::serialize(compiled_model.get_runtime_model(), "./model_fc_compile.xml", "./model_fc_compile.bin");
        
        // -------- Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Prepare input --------
        auto input_generate = ov::test::utils::InputGenerateData(0, 5);
        auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(), infer_request.get_input_tensor().get_shape(), input_generate);
        std::cout << "\n" << "input_tensor: ";
        for (size_t i = 0; i < tensor.get_size(); i++) {
            std::cout << tensor.data<float>()[i] << ", ";
        }
        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        std::cout << "\n" << "output_tensor: " << output_tensor.get_shape() << std::endl;
        for (size_t i = 0; i < output_tensor.get_size(); i++) {
            std::cout << output_tensor.data<float>()[i] << ", ";
        }
    }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput3) {
    int w_rank{0}, w_size{0};
    if (getenv("ENABLE_CCL")) {  // only apply for dynamic models for now
        w_rank = cldnn::Messenger::getInstance().getRank();
        w_size = cldnn::Messenger::getInstance().getSize();
    }
    std::cout << "Apply TP rank " << w_rank << ", size: " << w_size << std::endl;
    // -------- Construct model
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    // auto weights = std::vector<float>{1, 2, 3, 4};
    std::vector<int64_t> weights(4, 2);
    auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, weights);
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

    auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);

    auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    ov::serialize(model,
                  std::string("./model_fc_rank_" + std::to_string(w_rank) + ".xml"),
                  std::string("./model_fc_rank_" + std::to_string(w_rank) + ".bin"));

    ov::pass::Manager manager;
    manager.register_pass<FullyConnectedSplitInput>();
    manager.run_passes(model);
    ov::serialize(model,
                  std::string("./model_fc_pass_rank_" + std::to_string(w_rank) + ".xml"),
                  std::string("./model_fc_pass_rank_" + std::to_string(w_rank) + ".bin"));

    // -------- Loading a model to the device --------
    ov::Core core;
    ov::CompiledModel compiled_model;
    if (w_rank == 0)
        compiled_model = core.compile_model(model, "GPU.0");
    else
        compiled_model = core.compile_model(model, "GPU.1");
    ov::serialize(compiled_model.get_runtime_model(), "./model_fc_compile.xml", "./model_fc_compile.bin");

    // -------- Create an infer request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Prepare input --------
    auto input_generate = ov::test::utils::InputGenerateData(0, 5);
    auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(),
                                                          infer_request.get_input_tensor().get_shape(),
                                                          input_generate);
    if (w_rank == 0) {
        std::cout << "\n"
                  << "input_tensor: ";
        for (size_t i = 0; i < tensor.get_size(); i++) {
            std::cout << tensor.data<float>()[i] << ", ";
        }
        }
        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        infer_request.infer();

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        std::cout << "\n"
                  << "output_tensor: ";
        for (size_t i = 0; i < output_tensor.get_size(); i++) {
            std::cout << output_tensor.data<float>()[i] << ", ";
        }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput4) {
    {
        // -------- Construct model
        unsigned long test_size = 1024;
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_size, test_size});
        std::vector<float> weights(test_size * test_size, 2);
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{test_size, test_size}, weights);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

        // -------- Loading a model to the device --------
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU.0");

        // -------- Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Prepare input --------
        auto input_generate = ov::test::utils::InputGenerateData(0, 5);
        auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(),
                                                              infer_request.get_input_tensor().get_shape(),
                                                              input_generate);
        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        auto start = high_resolution_clock::now();
        for (int iter = 0; iter < 10000; iter++) {
            infer_request.infer();
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        std::cout << "infere avg time: " << duration.count()/10000 << std::endl;

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        // std::cout << "\n"
        //           << "output_tensor: " << output_tensor.get_shape() << std::endl;
        // for (size_t i = 0; i < output_tensor.get_size(); i++) {
        //     std::cout << output_tensor.data<float>()[i] << ", ";
        // }
    }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput5) {
    int w_rank{0};
    if (getenv("ENABLE_CCL")) {  // only apply for dynamic models for now
        w_rank = cldnn::Messenger::getInstance().getRank();
    }
    // -------- Construct model
    unsigned long test_size = 1024;
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_size, test_size});
    std::vector<float> weights(test_size * test_size, 2);
    auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{test_size, test_size}, weights);
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

    ov::pass::Manager manager;
    manager.register_pass<FullyConnectedSplitInput>();
    manager.run_passes(model);

    // -------- Loading a model to the device --------
    ov::Core core;
    ov::CompiledModel compiled_model;
    if (w_rank == 0)
        compiled_model = core.compile_model(model, "GPU.0");
    else
        compiled_model = core.compile_model(model, "GPU.1");

    // -------- Create an infer request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Prepare input --------
    auto input_generate = ov::test::utils::InputGenerateData(0, 5);
    auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(),
                                                          infer_request.get_input_tensor().get_shape(),
                                                          input_generate);
    infer_request.set_input_tensor(tensor);

    // -------- Do inference synchronously --------
    auto start = high_resolution_clock::now();
    for (int iter = 0; iter < 10000; iter++) {
        infer_request.infer();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "infere time: " << duration.count()/10000 << std::endl;

    // -------- Process output
    auto output_tensor = infer_request.get_output_tensor();
    // std::cout << "\n"
    //           << "output_tensor: " << output_tensor.get_shape() << std::endl;
    // for (size_t i = 0; i < output_tensor.get_size(); i++) {
    //     std::cout << output_tensor.data<float>()[i] << ", ";
    // }
}

TEST(TransformationTestsF1, FullyConnectedSplitInput6) {
    {
        // -------- Construct model
        unsigned long test_size = 1024;
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{test_size, test_size/2});
        std::vector<float> weights(test_size * (test_size/2), 2);
        auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{test_size, test_size/2}, weights);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});

        // -------- Loading a model to the device --------
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(model, "GPU.0");

        // -------- Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Prepare input --------
        auto input_generate = ov::test::utils::InputGenerateData(0, 5);
        auto tensor = ov::test::utils::create_and_fill_tensor(infer_request.get_input_tensor().get_element_type(),
                                                              infer_request.get_input_tensor().get_shape(),
                                                              input_generate);
        infer_request.set_input_tensor(tensor);

        // -------- Do inference synchronously --------
        auto start = high_resolution_clock::now();
        for (int iter = 0; iter < 10000; iter++) {
            infer_request.infer();
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        std::cout << "infere avg time: " << duration.count()/10000 << std::endl;

        // -------- Process output
        auto output_tensor = infer_request.get_output_tensor();
        // std::cout << "\n"
        //           << "output_tensor: " << output_tensor.get_shape() << std::endl;
        // for (size_t i = 0; i < output_tensor.get_size(); i++) {
        //     std::cout << output_tensor.data<float>()[i] << ", ";
        // }
    }
}