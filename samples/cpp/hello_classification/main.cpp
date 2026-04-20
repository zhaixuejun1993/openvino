// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

// Function to compare two tensors and calculate difference statistics
struct ComparisonResult {
    double max_abs_diff = 0.0;
    double avg_abs_diff = 0.0;
    double relative_error = 0.0;
    size_t total_elements = 0;
    bool results_match = false;
};

ComparisonResult compare_tensors(const ov::Tensor& tensor1, const ov::Tensor& tensor2, double tolerance = 1e-5) {
    ComparisonResult result;

    if (tensor1.get_shape() != tensor2.get_shape()) {
        std::cerr << "Error: Tensor shapes don't match!" << std::endl;
        return result;
    }

    if (tensor1.get_element_type() != ov::element::f32 || tensor2.get_element_type() != ov::element::f32) {
        std::cerr << "Error: Only FP32 tensors are supported for comparison!" << std::endl;
        return result;
    }

    const float* data1 = tensor1.data<const float>();
    const float* data2 = tensor2.data<const float>();

    result.total_elements = tensor1.get_size();
    double sum_abs_diff = 0.0;
    double sum_squares1 = 0.0;

    for (size_t i = 0; i < result.total_elements; ++i) {
        double diff = std::abs(static_cast<double>(data1[i]) - static_cast<double>(data2[i]));
        sum_abs_diff += diff;
        sum_squares1 += static_cast<double>(data1[i]) * static_cast<double>(data1[i]);

        if (diff > result.max_abs_diff) {
            result.max_abs_diff = diff;
        }
    }

    result.avg_abs_diff = sum_abs_diff / result.total_elements;
    result.relative_error = sum_squares1 > 0 ? std::sqrt(sum_abs_diff * sum_abs_diff / sum_squares1) : 0.0;
    result.results_match = result.max_abs_diff < tolerance;

    return result;
}

void print_shape(const ov::Shape& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]";
}

void print_first_elements(const ov::Tensor& tensor, size_t count = 10) {
    const float* data = tensor.data<const float>();
    size_t total = std::min(count, tensor.get_size());

    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < total; ++i) {
        std::cout << data[i];
        if (i < total - 1) std::cout << ", ";
    }
    if (total < tensor.get_size()) {
        std::cout << " ...";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string model_path = "/home/mcavus/xuejun/IR_naive.xml";

        if (argc > 1) {
            model_path = argv[1];
        }

        std::cout << "==========================================" << std::endl;
        std::cout << "OpenVINO CPU vs GPU Comparison Tool" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Model: " << model_path << std::endl << std::endl;

        // Initialize OpenVINO Runtime
        ov::Core core;

        // Read the model
        std::cout << "Loading model..." << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        // Print model information
        std::cout << "\nModel Information:" << std::endl;
        std::cout << "  Inputs: " << model->inputs().size() << std::endl;
        for (const auto& input : model->inputs()) {
            std::cout << "    - Name: " << input.get_any_name() << ", Shape: ";
            print_shape(input.get_shape());
            std::cout << ", Type: " << input.get_element_type() << std::endl;
        }

        std::cout << "  Outputs: " << model->outputs().size() << std::endl;
        size_t output_idx = 0;
        for (const auto& output : model->outputs()) {
            std::string output_name = output.get_names().empty() ?
                ("output_" + std::to_string(output_idx)) : output.get_any_name();
            std::cout << "    - Name: " << output_name << ", Shape: ";
            print_shape(output.get_shape());
            std::cout << ", Type: " << output.get_element_type() << std::endl;
            output_idx++;
        }
        std::cout << std::endl;

        // Compile model for CPU
        std::cout << "Compiling model for CPU..." << std::endl;
        ov::CompiledModel compiled_model_cpu = core.compile_model(model, "CPU");

        // Compile model for GPU
        std::cout << "Compiling model for GPU..." << std::endl;
        ov::CompiledModel compiled_model_gpu;
        bool gpu_available = false;

        try {
            compiled_model_gpu = core.compile_model(model, "GPU");
            gpu_available = true;
            std::cout << "GPU compilation successful!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "GPU not available or compilation failed: " << e.what() << std::endl;
            std::cout << "Will compare CPU results with itself for demonstration." << std::endl;
            compiled_model_gpu = core.compile_model(model, "CPU");
        }

        // Create inference requests
        ov::InferRequest infer_request_cpu = compiled_model_cpu.create_infer_request();
        ov::InferRequest infer_request_gpu = compiled_model_gpu.create_infer_request();

        // Prepare input data (random data for testing)
        std::cout << "\nPreparing input data..." << std::endl;
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (const auto& input : model->inputs()) {
            ov::Shape input_shape = input.get_shape();
            ov::Tensor input_tensor(input.get_element_type(), input_shape);

            // Fill with random data
            float* data = input_tensor.data<float>();
            for (size_t i = 0; i < input_tensor.get_size(); ++i) {
                data[i] = dist(gen);
            }

            std::cout << "  Input '" << input.get_any_name() << "' filled with random data" << std::endl;
            std::cout << "    First 10 values: ";
            print_first_elements(input_tensor, 10);

            // Set the same input for both devices
            infer_request_cpu.set_tensor(input.get_any_name(), input_tensor);
            infer_request_gpu.set_tensor(input.get_any_name(), input_tensor);
        }

        // Run inference on CPU
        std::cout << "\nRunning inference on CPU..." << std::endl;
        auto start_cpu = std::chrono::high_resolution_clock::now();
        infer_request_cpu.infer();
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
        std::cout << "  CPU inference time: " << duration_cpu.count() / 1000.0 << " ms" << std::endl;

        // Run inference on GPU
        std::cout << "\nRunning inference on " << (gpu_available ? "GPU" : "CPU (fallback)") << "..." << std::endl;
        auto start_gpu = std::chrono::high_resolution_clock::now();
        infer_request_gpu.infer();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
        std::cout << "  " << (gpu_available ? "GPU" : "CPU (fallback)")
                  << " inference time: " << duration_gpu.count() / 1000.0 << " ms" << std::endl;

        // Compare results
        std::cout << "\n==========================================" << std::endl;
        std::cout << "Results Comparison:" << std::endl;
        std::cout << "==========================================" << std::endl;

        bool all_match = true;
        output_idx = 0;
        for (const auto& output : model->outputs()) {
            const ov::Tensor& output_cpu = infer_request_cpu.get_output_tensor(output_idx);
            const ov::Tensor& output_gpu = infer_request_gpu.get_output_tensor(output_idx);

            std::string output_name = output.get_names().empty() ?
                ("output_" + std::to_string(output_idx)) : output.get_any_name();
            std::cout << "\nOutput: " << output_name << std::endl;
            std::cout << "  Shape: ";
            print_shape(output_cpu.get_shape());
            std::cout << std::endl;

            std::cout << "  CPU first 10 values: ";
            print_first_elements(output_cpu, 60);

            std::cout << "  " << (gpu_available ? "GPU" : "CPU") << " first 10 values: ";
            print_first_elements(output_gpu, 60);

            ComparisonResult comp = compare_tensors(output_cpu, output_gpu);

            std::cout << "\n  Comparison Statistics:" << std::endl;
            std::cout << "    Total elements: " << comp.total_elements << std::endl;
            std::cout << "    Max absolute difference: " << std::scientific << comp.max_abs_diff << std::endl;
            std::cout << "    Average absolute difference: " << comp.avg_abs_diff << std::endl;
            std::cout << "    Relative error: " << comp.relative_error << std::endl;
            std::cout << "    Results match (tolerance=1e-5): "
                      << (comp.results_match ? "YES ✓" : "NO ✗") << std::endl;

            if (!comp.results_match) {
                all_match = false;
            }
            output_idx++;
        }

        std::cout << "\n==========================================" << std::endl;
        if (all_match) {
            std::cout << "✓ SUCCESS: CPU and " << (gpu_available ? "GPU" : "CPU")
                      << " results match!" << std::endl;
        } else {
            std::cout << "✗ WARNING: CPU and " << (gpu_available ? "GPU" : "CPU")
                      << " results differ!" << std::endl;
        }
        std::cout << "==========================================" << std::endl;

        return all_match ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (const std::exception& ex) {
        std::cerr << "\nError: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
