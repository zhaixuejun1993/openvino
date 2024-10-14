// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define CL_VERSION_3_0 1
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <thread>
#include <algorithm>
#include <condition_variable>
#include <mutex>

#include "impls/registry/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "register.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "sync_tensor_inst.h"

namespace cldnn {
namespace ocl {

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
static std::map<int, std::string> oclErrorCode = {
    {0, "CL_SUCCESS"},
    {-1, "CL_DEVICE_NOT_FOUND"},
    {-2, "CL_DEVICE_NOT_AVAILABLE"},
    {-3, "CL_COMPILER_NOT_AVAILABLE"},
    {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {-5, "CL_OUT_OF_RESOURCES"},
    {-6, "CL_OUT_OF_HOST_MEMORY"},
    {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {-8, "CL_MEM_COPY_OVERLAP"},
    {-9, "CL_IMAGE_FORMAT_MISMATCH"},
    {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {-11, "CL_BUILD_PROGRAM_FAILURE"},
    {-12, "CL_MAP_FAILURE"},
    {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {-15, "CL_COMPILE_PROGRAM_FAILURE"},
    {-16, "CL_LINKER_NOT_AVAILABLE"},
    {-17, "CL_LINK_PROGRAM_FAILURE"},
    {-18, "CL_DEVICE_PARTITION_FAILED"},
    {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {-30, "CL_INVALID_VALUE"},
    {-31, "CL_INVALID_DEVICE_TYPE"},
    {-32, "CL_INVALID_PLATFORM"},
    {-33, "CL_INVALID_DEVICE"},
    {-34, "CL_INVALID_CONTEXT"},
    {-35, "CL_INVALID_QUEUE_PROPERTIES"},
    {-36, "CL_INVALID_COMMAND_QUEUE"},
    {-37, "CL_INVALID_HOST_PTR"},
    {-38, "CL_INVALID_MEM_OBJECT"},
    {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {-40, "CL_INVALID_IMAGE_SIZE"},
    {-41, "CL_INVALID_SAMPLER"},
    {-42, "CL_INVALID_BINARY"},
    {-43, "CL_INVALID_BUILD_OPTIONS"},
    {-44, "CL_INVALID_PROGRAM"},
    {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {-46, "CL_INVALID_KERNEL_NAME"},
    {-47, "CL_INVALID_KERNEL_DEFINITION"},
    {-48, "CL_INVALID_KERNEL"},
    {-49, "CL_INVALID_ARG_INDEX"},
    {-50, "CL_INVALID_ARG_VALUE"},
    {-51, "CL_INVALID_ARG_SIZE"},
    {-52, "CL_INVALID_KERNEL_ARGS"},
    {-53, "CL_INVALID_WORK_DIMENSION"},
    {-54, "CL_INVALID_WORK_GROUP_SIZE"},
    {-55, "CL_INVALID_WORK_ITEM_SIZE"},
    {-56, "CL_INVALID_GLOBAL_OFFSET"},
    {-57, "CL_INVALID_EVENT_WAIT_LIST"},
    {-58, "CL_INVALID_EVENT"},
    {-59, "CL_INVALID_OPERATION"},
    {-60, "CL_INVALID_GL_OBJECT"},
    {-61, "CL_INVALID_BUFFER_SIZE"},
    {-62, "CL_INVALID_MIP_LEVEL"},
    {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {-64, "CL_INVALID_PROPERTY"},
    {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {-66, "CL_INVALID_COMPILER_OPTIONS"},
    {-67, "CL_INVALID_LINKER_OPTIONS"},
    {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {-69, "CL_INVALID_PIPE_SIZE"},
    {-70, "CL_INVALID_DEVICE_QUEUE"},
    {-71, "CL_INVALID_SPEC_ID"},
    {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"},
};
#define CHECK_OCL_ERROR(err, msg)                                                                            \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
    }

static bool debug_enable = false;
static std::mutex debug_mutex;
static const std::chrono::_V2::system_clock::time_point perf_dump_start() {
    return std::chrono::high_resolution_clock::now();
}

static void perf_dump_done(const std::chrono::_V2::system_clock::time_point& start,
                           std::string str,
                           bool enable = false) {
    static std::once_flag check_flag;
    std::call_once(check_flag, [] {
        const char* env = getenv("OV_TP_P2P_DEBUG");
        if (env)
            debug_enable = true;
    });
    if (enable && debug_enable) {
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end - start;
        {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << str << " cost: " << elapsed_1.count() << " ms" << std::endl;
        }
    }
}

class gpu_semaphore {
public:
    gpu_semaphore(int count = 1) : count_(count), total_(count) {}
    void signal() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (count_ < total_)
            ++count_;
        cv_.notify_one();
    }
    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return count_ > 0;
        });
        --count_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
    int total_;
};

static gpu_semaphore gpu_lock;
class tensor_concat_memory {
public:
    tensor_concat_memory() : buf(nullptr), width(0), height(0), type(ov::element::f16) {}
    tensor_concat_memory(cl_mem _buf, size_t _w, size_t _h, size_t _stride, ov::element::Type _type)
        : buf(_buf),
          width(_w),
          height(_h),
          stride(_stride),
          type(_type) {}
    tensor_concat_memory(tensor_concat_memory& other) {
        buf = other.buf;
        width = other.width;
        height = other.height;
        stride = other.stride;
        type = other.type;
    }
    bool operator==(const tensor_concat_memory& other) const {
        return width == other.height && height == other.height && stride == other.stride;
    }

    void print() const {
        size_t data_size = 0;
        auto err = clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
        std::cout << "width = " << width << ", height = " << height << ", stride = " << stride
                  << ", type = " << type.to_string() << " -- actual_size = " << data_size << std::endl;
    }

    cl_mem buf;
    size_t width;
    size_t height;
    size_t stride;
    ov::element::Type type;
};

class simple_tensor_concat {
public:
    simple_tensor_concat() {}
    ~simple_tensor_concat() {}

    bool validate(std::shared_ptr<tensor_concat_memory>& src, std::shared_ptr<tensor_concat_memory>& dst) {
        if (!src->buf)
            return false;
        if (src->type != dst->type)
            return false;

        if (!dst->buf)
            return false;

        concat_mode = -1;
        if (src->width == dst->width) {
            // Vertical concat
            concat_mode = 0;
        } else if (src->height <= dst->height) {  // fake alignment issue
            // Horizontal concat
            concat_mode = 1;
        } else {
            return false;
        }
        return true;
    }

    cldnn::event::ptr concat(cldnn::stream& stream,
                             std::shared_ptr<tensor_concat_memory>& src,
                             std::shared_ptr<tensor_concat_memory>& dst,
                             size_t w_rank,
                             bool blocked = true) {
        const auto start = perf_dump_start();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();

        if (!validate(src, dst)) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            print(src, dst);
            std::cout << "simple_tensor_concat::validate failed due to src/dst mismatch." << std::endl;
        }

        size_t src_rec[3] = {0, 0, 0};
        size_t dst_rec[3] = {0, 0, 0};
        size_t rect[3] = {src->width, src->height, 1};
        cl_event event;
        cldnn::event::ptr sync_event = nullptr;
        if (concat_mode == 0) {
            // Vertical concat
            dst_rec[1] = src->height * w_rank;
            auto ret = clEnqueueCopyBufferRect(queue,
                                               src->buf,
                                               dst->buf,
                                               src_rec,
                                               dst_rec,
                                               rect,
                                               src->stride,
                                               src->height * src->stride,
                                               dst->stride,
                                               dst->stride * dst->width,
                                               0,
                                               nullptr,
                                               &event);
            if (ret != CL_SUCCESS) {
                std::cout << "0.clEnqueueCopyBufferRect failed: " << oclErrorCode[ret] << ", w_rank = " << w_rank
                          << std::endl;
                OPENVINO_THROW("0.clEnqueueCopyBufferRect failed: ", oclErrorCode[ret], ", w_rank = ", w_rank);
            }
        } else if (concat_mode == 1) {
            // Horizontal concat
            dst_rec[0] = src->width * w_rank;
            auto ret = clEnqueueCopyBufferRect(queue,
                                               src->buf,
                                               dst->buf,
                                               src_rec,
                                               dst_rec,
                                               rect,
                                               src->stride,
                                               src->height * src->stride,
                                               dst->stride,
                                               dst->stride * dst->width,
                                               0,
                                               nullptr,
                                               &event);
            if (ret != CL_SUCCESS) {
                std::cout << "1.clEnqueueCopyBufferRect failed: " << oclErrorCode[ret] << ", w_rank = " << w_rank
                          << std::endl;
                OPENVINO_THROW("1.clEnqueueCopyBufferRect failed: ", oclErrorCode[ret], ", w_rank = ", w_rank);
            }
        } else {
            std::cout << "tensor_concat failed: incorrect concat mode!" << std::endl;
            OPENVINO_THROW("tensor_concat failed: incorrect concat mode!");
        }
        if (blocked) {
            auto ret = clWaitForEvents(1, &event);
            CHECK_OCL_ERROR(ret, "clWaitForEvents failed");
            clReleaseEvent(event);
        } else {
            sync_event = ocl_stream.create_event(cl::Event(event));
        }

        perf_dump_done(start, std::string("tensor_concat"), false);
        return sync_event;
    }

    void print(const std::shared_ptr<tensor_concat_memory>& src, const std::shared_ptr<tensor_concat_memory>& dst) {
        std::cout << " src[]: ";
        src->print();

        std::cout << " dst[0]: ";
        dst->print();
        std::cout << std::endl;
    }

private:
    // 0 - vertical concat
    // 1 - horizontal concat
    int concat_mode;
};

class gpu_p2p_helper {
public:
    gpu_p2p_helper() {}
    ~gpu_p2p_helper() {}

    uint64_t derive_handle(cl_mem clbuf) {
        cl_int err;
        uint64_t fd;
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(fd), &fd, NULL);
        CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_ALLOCATION_HANDLE_INTEL failed");
        return fd;
    }

    cl_mem map_remote_mem(cl_context context, cl_mem clbuf, size_t size) {
        cl_int err;
        const auto start = perf_dump_start();
        uint64_t fd = derive_handle(clbuf);
        // Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        if (err < 0) {
            OPENVINO_ASSERT(false,
                            "clCreateBufferWithProperties failed, clbuf = %p, fd = %ld, size = %ld, new_cl_mem = %p\n",
                            clbuf,
                            fd,
                            size,
                            extMemBuffer);
        }

        perf_dump_done(start, std::string("derive_map_remote_mem host time"));
        return extMemBuffer;
    }

    cl_mem map_remote_mem(cl_context context, uint64_t fd, size_t size) {
        cl_int err;
        const auto start = perf_dump_start();
        // Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        CHECK_OCL_ERROR(err, "clCreateBufferWithProperties - CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR failed");

        perf_dump_done(start, std::string("map_remote_mem host time"));
        return extMemBuffer;
    }

    void destory_remote_mem(cl_mem clbuf) {
        clReleaseMemObject(clbuf);
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

    void remote_copy(cldnn::stream& stream, cl_mem src, cl_mem dst, size_t size) {
        const auto start = perf_dump_start();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 0, NULL, &ret);
        clWaitForEvents(1, &ret);  // blocked copy
        clReleaseEvent(ret);
        perf_dump_done(start,
                       std::string("p2p copy host time for ") + std::to_string(size) + std::string(" bytes"),
                       true);
        return;
    }

    std::shared_ptr<tensor_concat_memory> create_concat_mem(cl_mem src, cldnn::layout src_layout) {
        ov::element::Type element_type = src_layout.data_type;
        auto element_size = element_type.size();
        auto src_shape = src_layout.get_shape();
        auto src_width = src_shape[-1] * element_size;
        auto src_stride = src_shape[-1] * element_size;  // No pad
        auto src_height = ov::shape_size(src_shape) / src_shape[-1];
        return std::make_shared<tensor_concat_memory>(src, src_width, src_height, src_stride, element_type);
    }

    cldnn::event::ptr remote_copy_rect(cldnn::stream& stream,
                                       cl_mem src,
                                       cldnn::layout src_layout,
                                       cl_mem dst,
                                       cldnn::layout dst_layout,
                                       size_t w_rank,
                                       bool blocked) {
        const auto start = perf_dump_start();
        auto concat = std::make_shared<simple_tensor_concat>();
        auto mem_src = create_concat_mem(src, src_layout);
        auto mem_dst = create_concat_mem(dst, dst_layout);
        auto ret = concat->concat(stream, mem_src, mem_dst, w_rank, blocked);
        perf_dump_done(start,
                       std::string("p2p copy_rect host time for ") + std::to_string(src_layout.bytes_count()) +
                           std::string(" bytes"),
                       true);
        return ret;
    }
};

static void dump_cl_buf(cl_command_queue queue, cl_mem clbuf, size_t count, size_t offset) {
    cl_int err;
    std::vector<float> outBuf(count, 0);
    err = clEnqueueReadBuffer(queue, clbuf, CL_TRUE, offset, count * 4, outBuf.data(), 0, NULL, NULL);
    CHECK_OCL_ERROR(err, "clEnqueueReadBuffer failed");
    clFinish(queue);

    // std::cout << "The first " << count << "elements in cl_mem = " << clbuf << " are: " << std::endl;
    for (int i = 0; i < static_cast<int>(count); i++) {
        // printf("%f, ", outBuf[i]);
        std::cout << outBuf[i] << ", ";
        if (i && i % 16 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

class simple_tensor_add {
public:
    simple_tensor_add() {}
    ~simple_tensor_add() {
        for (auto& item : kernels) {
            if (item.second)
                clReleaseKernel(item.second);
        }
        kernels.clear();
        if (program)
            clReleaseProgram(program);
    }

    typedef enum _kernel_data_type {
        e_type_fp16 = 0,
        e_type_int8 = 1,
        e_type_fp32 = 2,
    } kernel_data_type;

    kernel_data_type element_type_to_kernel_data_type(ov::element::Type_t element_type) {
        switch (element_type) {
        case ov::element::f16:
            return kernel_data_type::e_type_fp16;
        case ov::element::i8:
            return kernel_data_type::e_type_int8;
        case ov::element::f32:
            return kernel_data_type::e_type_fp32;
        default:
            OPENVINO_THROW("Error: unsupported element type for kernel adder - ",
                           ov::element::Type(element_type).to_string().c_str());
            break;
        }
        return kernel_data_type::e_type_int8;
    }

    cl_kernel create_kernel(cldnn::stream& stream, const char* kernel_code, const char* kernelName) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        std::cout << "create_kernel: name = " << kernelName << std::endl;

        cl_uint knlcount = 1;
        const char* knlstrList[] = {kernel_code};
        size_t knlsizeList[] = {strlen(kernel_code)};

        cl_context context = ocl_stream.get_engine().get_cl_context().get();
        program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);
        CHECK_OCL_ERROR(err, "clCreateProgramWithSource failed");

        std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
        err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
        if (err < 0) {
            size_t logsize = 0;
            auto device = ocl_stream.get_engine().get_cl_device().get();
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
            CHECK_OCL_ERROR(err, "clGetProgramBuildInfo failed");

            std::vector<char> logbuf(logsize + 1, 0);
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
            std::cout << "clGetProgramBuildInfo failed: " << logbuf.data() << std::endl;
            // OPENVINO_ASSERT(err >= 0, "clGetProgramBuildInfo: ", logbuf.data());
        }
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        CHECK_OCL_ERROR(err, "clCreateKernel failed");
        return kernel;
    }

    cl_kernel get_or_create_kernel_if_possible(cldnn::stream& stream, kernel_data_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = kernels.find(type);
        if (it != kernels.end()) {
            // std::cout << "get_kernel: type = " << static_cast<int>(type) << std::endl;
            return it->second;
        }
        #define ADD_OP_KERNEL_SOURCE_CODE(DATA_TYPE)                                                                       \
            "kernel void tensor_add_kernel_" #DATA_TYPE "(const global " #DATA_TYPE " *src, global " #DATA_TYPE " *dst) {" \
            "const int id = get_global_id(0);"                                                                             \
            "dst[id] += src[id];"                                                                                          \
            "}"
        if (type == kernel_data_type::e_type_fp16) {
            const char tensor_add_kernel_fp16[] = ADD_OP_KERNEL_SOURCE_CODE(half);
            const char kernel_name[] = "tensor_add_kernel_half";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp16, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_int8) {
            const char tensor_add_kernel_int8[] = ADD_OP_KERNEL_SOURCE_CODE(char);
            const char kernel_name[] = "tensor_add_kernel_char";
            kernels[type] = create_kernel(stream, tensor_add_kernel_int8, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_fp32) {
            const char tensor_add_kernel_fp32[] = ADD_OP_KERNEL_SOURCE_CODE(float);
            const char kernel_name[] = "tensor_add_kernel_float";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp32, kernel_name);
            return kernels[type];
        } else {
            std::cout << "error: unsupported adder kernel data type " << static_cast<int>(type) << std::endl;
            // OPENVINO_THROW("error: unsupported adder kernel data type ", static_cast<int>(type));
        }
        #undef ADD_OP_KERNEL_SOURCE_CODE
        return kernels[type];
    }

    cl_kernel get_or_create_kernel_if_possible_sub(cldnn::stream& stream, kernel_data_type type, size_t width, size_t width_sub, size_t offset) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = kernels.find(type);
        if (it != kernels.end()) {
            // std::cout << "get_kernel: type = " << static_cast<int>(type) << std::endl;
            return it->second;
        }
#define ADD_OP_KERNEL_SOURCE_CODE(DATA_TYPE)                                                            \
    "kernel void tensor_add_kernel_" #DATA_TYPE "(const global " #DATA_TYPE " *src, global " #DATA_TYPE \
    " *dst, int width, int width_sub, int offset) {"                                   \
    "const int id = get_global_id(0);"                                                                  \
    "const int idx = id + offset;"                              \
    "dst[idx] += src[idx];"                                                   \
    "}"
        if (type == kernel_data_type::e_type_fp16) {
            const char tensor_add_kernel_fp16[] = ADD_OP_KERNEL_SOURCE_CODE(half);
            const char kernel_name[] = "tensor_add_kernel_half";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp16, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_int8) {
            const char tensor_add_kernel_int8[] = ADD_OP_KERNEL_SOURCE_CODE(char);
            const char kernel_name[] = "tensor_add_kernel_char";
            kernels[type] = create_kernel(stream, tensor_add_kernel_int8, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_fp32) {
            const char tensor_add_kernel_fp32[] = ADD_OP_KERNEL_SOURCE_CODE(float);
            const char kernel_name[] = "tensor_add_kernel_float";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp32, kernel_name);
            return kernels[type];
        } else {
            std::cout << "error: unsupported adder kernel data type " << static_cast<int>(type) << std::endl;
            // OPENVINO_THROW("error: unsupported adder kernel data type ", static_cast<int>(type));
        }
#undef ADD_OP_KERNEL_SOURCE_CODE
        return kernels[type];
    }

    event::ptr tensor_add(cldnn::stream& stream,
                          cl_mem src,
                          cl_mem dst,
                          size_t element_count,
                          kernel_data_type data_type) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        if (src == nullptr || dst == nullptr) {
            std::cout << "tensor_add: invalid arguments!" << std::endl;
        }
        OPENVINO_ASSERT(src != nullptr && dst != nullptr, "tensor_add: invalid arguments!");

        const auto start = perf_dump_start();
        cl_kernel kernel = get_or_create_kernel_if_possible(stream, data_type);
        perf_dump_done(start, std::string("get_or_create_kernel_if_possible"), false);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        CHECK_OCL_ERROR(err, "clSetKernelArg src failed");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &ret);
        CHECK_OCL_ERROR(err, "clEnqueueNDRangeKernel failed");
        // clWaitForEvents(1, &ret);

        perf_dump_done(start, std::string("tensor add host time"), false);
        return ocl_stream.create_event(cl::Event(ret));
    }

    event::ptr tensor_add_sub(cldnn::stream& stream,
                              cl_mem src,
                              cl_mem dst,
                              size_t element_count,
                              kernel_data_type data_type,
                              size_t width,
                              size_t width_sub,
                              size_t offset) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        if (src == nullptr || dst == nullptr) {
            std::cout << "tensor_add: invalid arguments!" << std::endl;
        }
        OPENVINO_ASSERT(src != nullptr && dst != nullptr, "tensor_add: invalid arguments!");

        const auto start = perf_dump_start();
        cl_kernel kernel = get_or_create_kernel_if_possible_sub(stream,
                                                                data_type,
                                                                static_cast<int>(width),
                                                                static_cast<int>(width_sub),
                                                                static_cast<int>(offset));
        perf_dump_done(start, std::string("get_or_create_kernel_if_possible"), false);

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        CHECK_OCL_ERROR(err, "clSetKernelArg src failed");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        err = clSetKernelArg(kernel, 2, sizeof(int), &width);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        err = clSetKernelArg(kernel, 3, sizeof(int), &width_sub);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        err = clSetKernelArg(kernel, 4, sizeof(int), &offset);
        CHECK_OCL_ERROR(err, "clSetKernelArg dst failed");

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &ret);
        CHECK_OCL_ERROR(err, "clEnqueueNDRangeKernel failed");
        // clWaitForEvents(1, &ret);

        perf_dump_done(start, std::string("tensor add host time"), false);
        return ocl_stream.create_event(cl::Event(ret));
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

private:
    cl_program program;
    std::mutex mutex;
    std::map<kernel_data_type, cl_kernel> kernels;
};

static gpu_p2p_helper& get_p2p_instance() {
    static gpu_p2p_helper gpu_p2p_instance;
    return gpu_p2p_instance;
}

static simple_tensor_add& get_adder_instance(size_t idx) {
    static simple_tensor_add adder_instance[4];
    return adder_instance[idx];
}

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<sync_tensor_impl>(*this);
    }

    sync_tensor_impl() : parent() {}

    ~sync_tensor_impl() {
        for (auto& mem : all_gather_remote_dst) {
            if (mem) {
                release_remote_mems(static_cast<cl_mem>(mem));
            }
        }
        all_gather_remote_dst.clear();
        for (auto& mem : all_reduce_remote_dst) {
            if (mem) {
                release_remote_mems(static_cast<cl_mem>(mem));
            }
        }
        all_reduce_remote_dst.clear();
    }

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

    void wait_p2p_done(cldnn::stream& stream,
                       cldnn::ocl::gpu_p2p_helper& p2p_helper,
                       ov::intel_gpu::SubMemoryManager::ptr& sub_mem_mgr,
                       int id,
                       size_t w_size,
                       int32_t w_rank,
                       int all_reduce_solution = 0,
                       bool validate = true) {
        // Wait for P2P transferred data are ready
        std::vector<int> copy_list(w_size, 1);
        copy_list[w_rank] = 0;
        auto start = perf_dump_start();

        // Wait P2P done
        while (true) {
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank) && copy_list[idx]) {
                    auto& remote_ocl_stream =
                        downcast<ocl::ocl_stream>(*sub_mem_mgr->_memorys_table[id][idx].stream_ptr);
                    cldnn::event::ptr event = nullptr;
                    if (all_reduce_solution == 1) {
                        event = sub_mem_mgr->_memorys_table[id][idx].events[0];
                    } else {
                        event = sub_mem_mgr->_memorys_table[id][w_rank].events[idx];
                    }
                    if (event) {
                        event->wait();
                        remote_ocl_stream.finish();
                        copy_list[idx] = 0;
                        // std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                        if (all_reduce_solution == 1)
                            sub_mem_mgr->_memorys_table[id][idx].events[0] = nullptr;
                        else
                            sub_mem_mgr->_memorys_table[id][w_rank].events[idx] = nullptr;
                        // MUST release remote cl_mem, but it will cause remote map failed.
                        // cl_mem remote_mem =
                        // static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][idx].remote_mem[w_rank]);
                        // clReleaseMemObject(remote_mem);
                    }
                }
            }
            auto left_size = std::accumulate(copy_list.begin(), copy_list.end(), 0);
            if (left_size == 0)
                break;
            auto end = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end - start;
            if (duration.count() > 10000) {
                start = perf_dump_start();
                std::cout << "rank[" << w_rank << "]Error: sync_tensor wait_p2p_done timeout..." << std::endl;
            }
        }
        perf_dump_done(start,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait p2p done"),
                       true);
        return;
    }

    void print_internal_buffer(sync_tensor_inst& instance,
                               std::vector<cldnn::memory::ptr>& bufs,
                               cldnn::layout& layout,
                               size_t w_size,
                               size_t w_rank) {
        for (size_t i = 0; i < w_size; i++) {
            std::cout << "\trank[" << w_rank << "]: bufs[" << i << "] = " << bufs[i]
                      << ", layout = " << layout.to_short_string();
            if (bufs[i]) {
                auto cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(bufs[i])->get_buffer().get();
                size_t data_size = 0;
                auto err = clGetMemObjectInfo(cl_buf, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
                CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
                std::cout << ", buf_size = " << data_size;
            }
            std::cout << std::endl;
        }
    }

    bool update_internal_buffer(sync_tensor_inst& instance,
                                std::vector<cldnn::memory::ptr>& bufs,
                                cldnn::layout& last_layout,
                                cldnn::layout& layout,
                                size_t w_size,
                                size_t w_rank,
                                int all_reduce_solution) {
        auto& engine = instance.get_network().get_engine();
        size_t required_size = layout.bytes_count();
        bool allocated = false;
        if (0) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << "Before update_internal_buffer: " << std::endl;
            print_internal_buffer(instance, bufs, layout, w_size, w_rank);
        }

        auto need_realloc = [&](size_t idx) {
            if (all_reduce_solution == 2) {
                if (idx == 0)
                    return false;
            } else {
                if (idx == w_rank)
                    return false;
            }

            if (bufs[idx] == nullptr || last_layout.bytes_count() == 0)
                return true;

            if (bufs[idx]->size() < required_size)
                return true;

            // Batch has been changed to smaller, need reallocate to decrease memory.
            auto last_batch = last_layout.batch() * last_layout.feature();
            auto batch = layout.batch() * layout.feature();
            if (last_batch > batch)
                return true;
            return false;
        };
        for (size_t i = 0; i < w_size; i++) {
            if (!need_realloc(i)) {
                continue;
            }
            // size_t origin_size = bufs[i] != nullptr ? bufs[i]->size() : 0;
            bufs[i] = nullptr;
            bufs[i] = engine.allocate_memory(layout, cldnn::allocation_type::cl_mem, false);
            allocated = true;
            if (0) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                std::cout << "tensor_sync allocate: rank[" << w_rank << "]: layout[" << i
                          << "]=" << layout.to_short_string() << ", last_layout[" << i
                          << "]=" << last_layout.to_short_string()
                          << std::endl;
            }
        }
        return allocated;
    }

    void release_remote_mems(cl_mem remote_mems) {
        if (remote_mems) {
            size_t data_size = 0;
            auto _cl_mem = static_cast<cl_mem>(remote_mems);
            if (0) {
                auto error = clGetMemObjectInfo(_cl_mem, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
                std::cout << "clReleaseMemObject...size = " << data_size << ", error: " << error << std::endl;
            }
            clReleaseMemObject(_cl_mem);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();
        const bool pass_through_events = false;

        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto is_all_reduce = instance.get_impl_params()->need_add == true;
        if (!is_all_reduce && all_gather_remote_dst.size() == 0) {
            all_gather_remote_dst.assign(w_size, nullptr);
        }
        all_reduce_remote_dst.assign(w_size, nullptr);

        int all_reduce_solution = 2;
        if (is_all_reduce) {
            // if (instance.get_output_memorys()[0]->size() > 1024*1024)
            //     all_reduce_solution = 2;
            // if (instance.get_output_memorys()[0]->size() > 1024*1024) {
            //     // all_reduce_solution = 2;
            //     std::lock_guard<std::mutex> lock(debug_mutex);
            //     std::cout << "[>>>Rank] " << w_rank << ", size > 1024x1024" << std::endl;
            // } else {
            //     std::lock_guard<std::mutex> lock(debug_mutex);
            //     std::cout << "[---Rank] " << w_rank << ", size < 1024x1024" << std::endl;
            // }
            // all_reduce_solution = 2;
            // std::cout << "[Rank] " << w_rank << ", all_reduce_solution: " << all_reduce_solution << std::endl;
            const char* all_reduce_add_solution = getenv("OV_TP_ALLREDUCE_ADD_solution");
            if (all_reduce_add_solution)
                all_reduce_solution = std::atoi(all_reduce_add_solution);
        }
        auto start = perf_dump_start();
        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        perf_dump_done(start,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait events"),
                       true);

        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto id = 0;
        // auto id = sub_mem_mgr->get_memory_id(w_rank);
        // sub_mem_mgr->_memorys_table[id][w_rank].layer_finished_flag = false;
        sub_mem_mgr->set_memory_used(id, w_rank);
        auto start_1 = perf_dump_start();
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
                for (size_t i = 0; i < w_size; i++) {
                    sub_mem_mgr->_memorys_table[id][i].flag = false;
                    for (size_t j = 0; j < w_size; j++) {
                        sub_mem_mgr->_memorys_table[id][i].events[j] = nullptr;
                        sub_mem_mgr->_memorys_table[id][i].recv_flag[j] = false;
                        sub_mem_mgr->_memorys_table[id][i].recv_flag_concat[j] = false;
                        sub_mem_mgr->_memorys_table[id][i].add_flag[j] = false;
                    }
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
            auto end_1 = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end_1 - start_1;
            if (duration.count() > 10000) {
                start_1 = perf_dump_start();
                std::cout << "rank[" << w_rank << "]Error: sync_tensor wait data ready timeout..." << std::endl;
            }
        }

        perf_dump_done(start_1,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait data ready"),
                       true);
        gpu_p2p_helper& gpu_p2p_instance = get_p2p_instance();
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto local_context = ocl_stream.get_engine().get_cl_context().get();
        auto p2p_src_layout = instance.get_output_layout(0);
        bool need_update_remote_mems = false;
        if (is_all_reduce) {
            OPENVINO_ASSERT(1 == instance.get_output_memorys().size(), "All reduce only has one output!");
            if (all_reduce_solution == 2)
                sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0] = instance.get_output_memorys()[0];
            else
                sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank] = instance.get_output_memorys()[0];
            sub_mem_mgr->_memorys_table[id][w_rank].output = instance.get_output_memorys()[0];
            // Allocate or reuse buffer for P2P target, same shape with output[0]
            p2p_src_layout = instance.get_output_layout(0);
            need_update_remote_mems = update_internal_buffer(instance,
                                                             sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs,
                                                             sub_mem_mgr->_memorys_table[id][w_rank].layout,
                                                             p2p_src_layout,
                                                             w_size,
                                                             w_rank,
                                                             all_reduce_solution);
        } else {
            OPENVINO_ASSERT(2 == instance.get_output_memorys().size(),
                            "All gather need additional buffer for concat result!");
            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank] = instance.get_output_memorys()[1];
            sub_mem_mgr->_memorys_table[id][w_rank].output = instance.get_output_memorys()[0];
            // All gather doesn't need intermediate buffer at all.
            p2p_src_layout = instance.get_output_layout(1);
            auto tmp =
                std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.get_output_memorys()[0])->get_buffer().get();
            if (tmp != all_gather_current_dst) {
                need_update_remote_mems = true;
                all_gather_current_dst = tmp;
            }
        }
        if (all_reduce_solution == 1)
            sub_mem_mgr->_memorys_table[id][w_rank].recv_flag[w_rank] = true;
        if (all_reduce_solution == 2) {
            sub_mem_mgr->_memorys_table[id][w_rank].recv_flag[0] = true;
            sub_mem_mgr->_memorys_table[id][w_rank].recv_flag_concat[0] = true;

        } else {
            sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        }

        // The mapped remote cl_mem will hold the original cl_mem, it should be released if the original cl_mem has been
        // released, else it will cause gpu memory leak.
        if (need_update_remote_mems) {
            if (debug_enable) {
                std::cout << "release_remote_mems: old_layout = "
                          << sub_mem_mgr->_memorys_table[id][w_rank].layout.to_short_string()
                          << ", new_layout = " << p2p_src_layout.to_short_string() << std::endl;
            }
        }

        std::vector<cldnn::event::ptr> sync_events;
        if (is_all_reduce && all_reduce_solution == 2) {
            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].recv_flag[0] == true)
                        wait_all_ouput_ready++;
                }
                if (wait_all_ouput_ready == w_size) {
                    break;
                }
            }
            auto split_parts = [](int len, int n) {
                int average = len / n;
                std::vector<int> parts(n, average);
                parts.back() = len - average * (n - 1);
                return parts;
            };

            auto output_layout = instance.get_output_layout(0);
            ov::element::Type output_element_type = output_layout.data_type;
            auto output_element_size = output_element_type.size();
            auto output_shape = output_layout.get_shape();
            auto sub_out_dim_vec = split_parts(output_shape[-1], w_size);
            if (0) {
                for (size_t iter = 0; iter < w_size; iter++) {
                    std::cout << "sub_out_dim_vec[" << iter << "] " << sub_out_dim_vec[iter] << std::endl;
                }
            }
            auto output_height = ov::shape_size(output_shape) / output_shape[-1];
            if (0)
            std::cout << "------------- Copy Start -------------" << std::endl;
            for (int32_t i = 0; i < static_cast<int>(w_size) - 1; i++) {
                auto src_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                    sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                auto src_buf = src_mem->get_buffer().get();

                int32_t sub_part = (w_rank - i) >= 0 ? (w_rank - i) : ((w_rank - i) + w_size);
                int32_t rec_sub_part = (sub_part - 1) >= 0 ? (sub_part - 1) : ((sub_part - 1) + w_size);
                auto dst_idx = (w_rank + 1) % w_size;

                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "[iter] " << i << ",[rank] " << w_rank << ", id: " << id << " ,dst_idx: " << dst_idx
                              << ", sub_part: " << sub_part << ", rec_sub_part: " << rec_sub_part << std::endl;
                }
                // auto dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].recv_bufs[i + 1];
                // size_t data_size = dst_mem->size();
                // auto dst_cl_buf_remote = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                // auto dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);

                cl_mem dst_cl_buf = nullptr;
                cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].recv_bufs[i + 1];
                size_t data_size = dst_mem->size();
                auto dst_cl_buf_remote = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems_p2p[i]);
                if (need_update_remote_mems || dst_cl_buf == nullptr) {
                    if (dst_cl_buf) {
                        release_remote_mems(dst_cl_buf);
                    }
                    dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    sub_mem_mgr->_memorys_table[id][w_rank].remote_mems_p2p[i] = dst_cl_buf;
                }

                auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
                auto queue = ocl_stream.get_cl_queue().get();
                int32_t off_set = 0;
                for (int32_t j = 0; j < sub_part; j++) {
                    off_set = off_set + sub_out_dim_vec[j];
                }
                size_t src_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                size_t dst_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                cl_event event;
                size_t rect[3] = {sub_out_dim_vec[sub_part] * output_element_size, output_height, 1};
                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "[iter] " << i << ",[rank] " << w_rank << std::endl;
                    std::cout << "[sub_part] " << sub_part << ",[rec_sub_part] " << rec_sub_part << \
                    ",sub_out_dim_vec[sub_part] " << sub_out_dim_vec[sub_part] << std::endl;
                    std::cout << "offset: " << off_set << ", output_element_size: " << output_element_size << std::endl;
                    std::cout << "src_rec: " << src_rec[0] << ", " << src_rec[1] << ", " << src_rec[2] << std::endl;
                    std::cout << "dst_rec: " << dst_rec[0] << ", " << dst_rec[1] << ", " << dst_rec[2] << std::endl;
                    std::cout << "rect: " << rect[0] << ", " << rect[1] << ", " << rect[2] << std::endl;
                }
                auto ret = clEnqueueCopyBufferRect(queue,
                                                   src_buf,
                                                   dst_cl_buf,
                                                   src_rec,
                                                   dst_rec,
                                                   rect,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   &event);
                if (ret != CL_SUCCESS) {
                    std::cout << "0.clEnqueueCopyBufferRect failed: " << oclErrorCode[ret] << ", idx = " << i
                              << std::endl;
                    OPENVINO_THROW("0.clEnqueueCopyBufferRect failed: ", oclErrorCode[ret], ", idx = ", i);
                }
                ret = clWaitForEvents(1, &event);
                CHECK_OCL_ERROR(ret, "clWaitForEvents failed");
                clReleaseEvent(event);

                sub_mem_mgr->_memorys_table[id][dst_idx].recv_flag[i + 1] = true;

                while (true) {
                    size_t wait_all_ouput_ready = 0;
                    for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                        if (sub_mem_mgr->_memorys_table[id][idx].recv_flag[i + 1] == true)
                            wait_all_ouput_ready++;
                    }
                    if (wait_all_ouput_ready == w_size)
                        break;
                }

                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "[After Copy Rank] " << w_rank << ", [iter] " << i << std::endl;
                    for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                        auto mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx]);
                        auto buf = mem->get_buffer().get();
                        dump_cl_buf(ocl_stream.get_cl_queue().get(), buf, mem->count(), 0);
                    }
                }

                auto dst_mem_add = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                    sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                auto dst_cl_buf_add = dst_mem_add->get_buffer().get();
                auto& adder_instance = get_adder_instance(w_rank);

                auto src_mem_add = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[i + 1];
                auto src_cl_buf_add = std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem_add)->get_buffer().get();

                sub_mem_mgr->_memorys_table[id][w_rank].last_rec_part = rec_sub_part;
                int32_t off_set_add = 0;
                for (int32_t j = 0; j < rec_sub_part; j++)
                    off_set_add = off_set_add + sub_out_dim_vec[j];

                cldnn::event::ptr sync_add_event;
                if (0)
                    std::cout << "[rank] " << w_rank
                              << ", element_count: " << output_height * sub_out_dim_vec[rec_sub_part]
                              << ", sub_width: " << sub_out_dim_vec[rec_sub_part]
                              << ", offset: " << output_height * sub_out_dim_vec[rec_sub_part]
                              << ", width: " << output_shape[-1] << std::endl;
                auto start_add = perf_dump_start();
                sync_add_event = adder_instance.tensor_add_sub(
                    stream,
                    src_cl_buf_add,
                    dst_cl_buf_add,
                    output_height * sub_out_dim_vec[rec_sub_part],
                    adder_instance.element_type_to_kernel_data_type(dst_mem_add->get_layout().data_type),
                    output_shape[-1],
                    sub_out_dim_vec[rec_sub_part],
                    output_height * off_set_add);
                sync_add_event->wait();
                auto end_add = perf_dump_start();
                std::chrono::duration<double, std::milli> duration = end_add - start_add;
                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "Add Solution[" << all_reduce_solution << "] " << "Rank[" << w_rank
                              << "] sync tensor p2p add total cost: " << duration.count() << " ms" << std::endl;
                }
                sub_mem_mgr->_memorys_table[id][w_rank].add_flag[i + 1] = true;

                while (true) {
                    size_t wait_all_ouput_ready = 0;
                    for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                        if (sub_mem_mgr->_memorys_table[id][idx].add_flag[i + 1] == true)
                            wait_all_ouput_ready++;
                    }
                    if (wait_all_ouput_ready == w_size)
                        break;
                }

                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "[After Add Rank] " << w_rank << std::endl;
                    for (int idx = 0; idx < 1; idx++) {
                        auto mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx]);
                        auto buf = mem->get_buffer().get();
                        dump_cl_buf(ocl_stream.get_cl_queue().get(), buf, mem->count(), 0);
                    }
                }
            }
            // bool need_update_flag = true;
            {
                auto dst_idx = (w_rank + 1) % w_size;
                cl_mem dst_cl_buf = nullptr;
                cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].output;
                size_t data_size = dst_mem->size();
                auto dst_cl_buf_remote = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                dst_cl_buf = static_cast<cl_mem>(all_reduce_remote_dst[dst_idx]);
                if (need_update_remote_mems || dst_cl_buf == nullptr) {
                    if (dst_cl_buf) {
                        release_remote_mems(dst_cl_buf);
                     }
                    dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    all_reduce_remote_dst[dst_idx] = dst_cl_buf;
                }
            }
            for (int32_t i = 0; i < static_cast<int>(w_size) - 1; i++) {
                auto src_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                    sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                auto src_buf = src_mem->get_buffer().get();

                auto dst_idx = (w_rank + 1) % w_size;

                // auto dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].recv_bufs[0];
                // size_t data_size = dst_mem->size();
                // auto dst_cl_buf_remote = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                // auto dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);

                // cl_mem dst_cl_buf = nullptr;
                // cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].output;
                // size_t data_size = dst_mem->size();
                // auto dst_cl_buf_remote = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                // dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);
                // if (need_update_remote_mems || dst_cl_buf == nullptr) {
                //     // if (dst_cl_buf) {
                //     //     release_remote_mems(dst_cl_buf);
                //     // }
                //     dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                //     sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0] = dst_cl_buf;
                // }
                cl_mem dst_cl_buf = nullptr;
                dst_cl_buf = static_cast<cl_mem>(all_reduce_remote_dst[dst_idx]);

                // if (dst_cl_buf == nullptr) {
                //     if (1) {
                //         std::lock_guard<std::mutex> lock(debug_mutex);
                //         std::cout << "[1][New][rank][S] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                //                   << ", data_size: " << data_size << std::endl;
                //     }
                //     dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                //     sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0] = dst_cl_buf;
                //     // need_update_flag = false;
                //     if (1) {
                //         std::lock_guard<std::mutex> lock(debug_mutex);
                //         std::cout << "[1][New][rank][D] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                //                   << ", dst_cl_buf: " << dst_cl_buf << std::endl;
                //     }
                // } else {
                //     if (1) {
                //         std::lock_guard<std::mutex> lock(debug_mutex);
                //         std::cout << "[3][Old][rank] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                //                   << ", dst_cl_buf: " << dst_cl_buf << std::endl;
                //     }
                    // if (need_update_remote_mems && need_update_flag) {
                    //     if (1) {
                    //         std::lock_guard<std::mutex> lock(debug_mutex);
                    //         std::cout << "[2][Rel][rank][S] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                    //                   << ", dst_cl_buf: " << dst_cl_buf << std::endl;
                    //     }
                    //     release_remote_mems(dst_cl_buf);
                    //     dst_cl_buf = nullptr;
                    //     if (1) {
                    //         std::lock_guard<std::mutex> lock(debug_mutex);
                    //         std::cout << "[2][Rel][rank][D] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                    //                   << ", dst_cl_buf: " << dst_cl_buf << std::endl;
                    //     }
                    //     if (1) {
                    //         std::lock_guard<std::mutex> lock(debug_mutex);
                    //         std::cout << "[2][New][rank][S] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                    //                   << std::endl;
                    //     }
                    //     dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    //     sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0] = dst_cl_buf;
                    //     need_update_flag = false;
                    //     if (1) {
                    //         std::lock_guard<std::mutex> lock(debug_mutex);
                    //         std::cout << "[2][New][rank][D] " << w_rank << ", iter: " << i << ", dst_idx: " << dst_idx
                    //                   << ", dst_cl_buf: " << dst_cl_buf << std::endl;
                    //     }
                    // }
                // }

                auto sub_part = sub_mem_mgr->_memorys_table[id][w_rank].last_rec_part;
                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << "[iter] " << i << ", [rank] " << w_rank << ", copy to idx: " << dst_idx
                              << ", rec_sub_part: " << sub_mem_mgr->_memorys_table[id][w_rank].last_rec_part
                              << std::endl;
                }
                auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
                auto queue = ocl_stream.get_cl_queue().get();
                int32_t off_set = 0;
                for (int32_t j = 0; j < sub_part; j++) {
                    off_set = off_set + sub_out_dim_vec[j];
                }
                size_t src_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                size_t dst_rec[3] = {off_set * output_element_size * output_height, 0, 0};
                cl_event event;
                size_t rect[3] = {sub_out_dim_vec[sub_part] * output_element_size, output_height, 1};
                auto ret = clEnqueueCopyBufferRect(queue,
                                                   src_buf,
                                                   dst_cl_buf,
                                                   src_rec,
                                                   dst_rec,
                                                   rect,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   &event);
                if (ret != CL_SUCCESS) {
                    std::cout << "0.clEnqueueCopyBufferRect failed: " << oclErrorCode[ret] << ", idx = " << i
                              << std::endl;
                    OPENVINO_THROW("0.clEnqueueCopyBufferRect failed: ", oclErrorCode[ret], ", idx = ", i);
                }
                ret = clWaitForEvents(1, &event);
                CHECK_OCL_ERROR(ret, "clWaitForEvents failed");
                clReleaseEvent(event);
                sub_mem_mgr->_memorys_table[id][dst_idx].last_rec_part = sub_part;

                sub_mem_mgr->_memorys_table[id][dst_idx].recv_flag_concat[i + 1] = true;

                while (true) {
                    size_t wait_all_ouput_ready = 0;
                    for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                        if (sub_mem_mgr->_memorys_table[id][idx].recv_flag_concat[i + 1] == true)
                            wait_all_ouput_ready++;
                    }
                    if (wait_all_ouput_ready == w_size)
                        break;
                }

                if (0) {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    std::cout << std::endl;
                    std::cout << "[Concat Rank] " << w_rank <<  " [iter] " << i << std::endl;
                    for (int idx = 0; idx < 1; idx++) {
                        auto mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx]);
                        auto buf = mem->get_buffer().get();
                        dump_cl_buf(ocl_stream.get_cl_queue().get(), buf, mem->count(), 0);
                    }
                }
            }

            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].flag == true)
                        wait_all_ouput_ready++;
                }
                if (wait_all_ouput_ready == w_size) {
                    break;
                }
            }

            // cl_mem dst_cl_buf = nullptr;
            // dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);
            // if (dst_cl_buf) {
            //     if (1) {
            //         std::lock_guard<std::mutex> lock(debug_mutex);
            //         std::cout << "[2][Rel][rank][S] " << w_rank << ", dst_cl_buf: " << dst_cl_buf << std::endl;
            //     }
            //     release_remote_mems(dst_cl_buf);
            //     sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0] = nullptr;
            //     if (1) {
            //         std::lock_guard<std::mutex> lock(debug_mutex);
            //         std::cout << "[2][Rel][rank][D] " << w_rank << ", dst_cl_buf: " << dst_cl_buf << std::endl;
            //     }
            // }

            if (0) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                std::cout << "[After Concat Rank] " << w_rank << std::endl;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    auto mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                        sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx]);
                    auto buf = mem->get_buffer().get();
                    dump_cl_buf(ocl_stream.get_cl_queue().get(), buf, mem->count(), 0);
                }
            }
        } else {
        std::vector<int> wait_list(w_size, 1);
        auto start_2 = perf_dump_start();
        wait_list[w_rank] = 0;  // no need to wait for itself
        size_t data_size = 0;
        event::ptr sync_event = nullptr;
        auto src_p2p_buf =
            std::dynamic_pointer_cast<const ocl::gpu_buffer>(sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank]);
        auto src_cl_buf = src_p2p_buf->get_buffer().get();
        if (all_reduce_solution == 1) {
            while (true) {
                if (w_rank == 0)
                    break;
                if (w_rank != 0 && sub_mem_mgr->_memorys_table[id][0].flag) {
                    cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][0].recv_bufs[w_rank];
                    auto dst_cl_buf_remote =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                    data_size = dst_mem->size();
                    auto dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    auto p2p_data_size = p2p_src_layout.bytes_count();

                    gpu_lock.acquire();
                    gpu_p2p_instance.remote_copy(stream, src_cl_buf, dst_cl_buf, p2p_data_size);
                    gpu_lock.signal();

                    sub_mem_mgr->_memorys_table[id][0].recv_flag[w_rank] = true;
                    {
                        std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                        sub_mem_mgr->_memorys_table[id][w_rank].events[0] = stream.create_user_event(true);
                    }
                    break;
                }
            }
        } else {
        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    cl_mem dst_cl_buf = nullptr;
                    // cldnn::memory::ptr dst_mem = nullptr;
                                        // if (is_all_reduce) {
                    //     dst_mem = sub_mem_mgr->_memorys_table[id][idx].recv_bufs[w_rank];
                    // } else {
                    //     dst_mem = sub_mem_mgr->_memorys_table[id][idx].output;
                    // }
                    // auto dst_cl_buf_remote =
                    //     std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();

                    // data_size = dst_mem->size();
                    // auto dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                    if (is_all_reduce) {
                        cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][idx].recv_bufs[w_rank];
                        data_size = dst_mem->size();
                        auto dst_cl_buf_remote =
                            std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                        dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[idx]);
                        if (need_update_remote_mems || dst_cl_buf == nullptr) {
                            if (dst_cl_buf) {
                                release_remote_mems(dst_cl_buf);
                            }
                            dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                            sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[idx] = dst_cl_buf;
                        }
                        // if (need_update_remote_mems || dst_cl_buf == nullptr) {
                        //     std::lock_guard<std::mutex> lock(debug_mutex);
                        //     std::cout << "[1][New][AR] " << "sub_mem_mgr->_memorys_table[id][" << w_rank << "].remote_mems["
                        //     << idx << "], dst_cl_buf:" << dst_cl_buf << std::endl;
                        // } else {
                        //     std::lock_guard<std::mutex> lock(debug_mutex);
                        //     // std::cout << "[1][Old][AR] " << "sub_mem_mgr->_memorys_table[id][" << w_rank << "].remote_mems["
                        //     // << idx << "], dst_cl_buf:" << dst_cl_buf << std::endl;
                        // }
                    } else {
                        cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][idx].output;
                        data_size = dst_mem->size();
                        auto dst_cl_buf_remote =
                            std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                        dst_cl_buf = static_cast<cl_mem>(all_gather_remote_dst[idx]);
                        if (need_update_remote_mems || dst_cl_buf == nullptr) {
                            if (dst_cl_buf) {
                                release_remote_mems(dst_cl_buf);
                            }
                            dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context, dst_cl_buf_remote, data_size);
                            all_gather_remote_dst[idx] = dst_cl_buf;
                        }
                    }
                    auto p2p_data_size = p2p_src_layout.bytes_count();
                    {
                        gpu_lock.acquire();
                        if (is_all_reduce) {
                            gpu_p2p_instance.remote_copy(stream, src_cl_buf, dst_cl_buf, p2p_data_size);
                        } else {
                            gpu_p2p_instance.remote_copy_rect(stream,
                                                              src_cl_buf,
                                                              instance.get_output_layout(1),
                                                              dst_cl_buf,
                                                              instance.get_output_layout(0),
                                                              w_rank,
                                                              false);
                        }
                        gpu_lock.signal();
                    }
                    // P2P has been done.
                    {
                        std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                        sub_mem_mgr->_memorys_table[id][idx].events[w_rank] = stream.create_user_event(true);
                    }
                    // gpu_p2p_instance.destory_remote_mem(dst_cl_buf);
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                break;
            }
            auto end_2 = perf_dump_start();
            std::chrono::duration<double, std::milli> duration = end_2 - start_2;
            if (duration.count() > 10000) {
                start_2 = perf_dump_start();
                std::cout << "rank[" << w_rank << "]Error: sync_tensor p2p write timeout..." << std::endl;
            }
        }
    }

        auto str_need_add = instance.get_impl_params()->need_add ? std::string("[need_add]") : std::string("");
        perf_dump_done(start_2,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor p2p write ") +
                           std::to_string(data_size) + " bytes" + str_need_add,
                       true);

        // P2P adopts sync write to avoid the problem of event cannot work across contexts
        if (all_reduce_solution == 1) {
            if (w_rank == 0)
                wait_p2p_done(stream, gpu_p2p_instance, sub_mem_mgr, id, w_size, w_rank, all_reduce_solution, false);
        } else {
            wait_p2p_done(stream, gpu_p2p_instance, sub_mem_mgr, id, w_size, w_rank, all_reduce_solution, false);
        }

        // std::vector<cldnn::event::ptr> sync_events;
        if (is_all_reduce) {
            if (all_reduce_solution == 1) {
                if (w_rank == 0) {
                    auto dst_mem_add = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(0));
                    auto dst_cl_buf_add = dst_mem_add->get_buffer().get();
                    auto& adder_instance = get_adder_instance(w_rank);
                    for (size_t idx = 0; idx < w_size; idx++) {
                        if (idx != 0) {
                            auto src_mem_add = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx];
                            auto src_cl_buf_add =
                                std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem_add)->get_buffer().get();
                            sync_event = adder_instance.tensor_add(
                                stream,
                                src_cl_buf_add,
                                dst_cl_buf_add,
                                dst_mem_add->count(),
                                adder_instance.element_type_to_kernel_data_type(dst_mem_add->get_layout().data_type));
                            sync_events.emplace_back(sync_event);
                        }
                    }
                    for (auto& evt : sync_events) {
                        evt->wait();
                    }
                    // broadcast the all-reduce result on rank=0, to the others
                    auto src_p2p_buf_broadcast =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(0));
                    auto src_cl_buf_broadcast = src_p2p_buf_broadcast->get_buffer().get();
                    for (size_t idx = 0; idx < w_size; idx++) {
                        if (idx != 0) {
                            cldnn::memory::ptr dst_mem_broadcast = sub_mem_mgr->_memorys_table[id][idx].recv_bufs[0];
                            auto dst_cl_buf_remote_broadcast =
                                std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem_broadcast)->get_buffer().get();
                            auto data_size_broadcast = dst_mem_broadcast->size();
                            auto dst_cl_buf_broadcast = gpu_p2p_instance.map_remote_mem(local_context,
                                                                                        dst_cl_buf_remote_broadcast,
                                                                                        data_size_broadcast);
                            auto p2p_data_size = p2p_src_layout.bytes_count();
                            gpu_p2p_instance.remote_copy(stream,
                                                         src_cl_buf_broadcast,
                                                         dst_cl_buf_broadcast,
                                                         p2p_data_size);
                            sub_mem_mgr->_memorys_table[id][idx].recv_flag[0] = true;
                        }
                    }
                } else {
                    while (true) {
                        if (sub_mem_mgr->_memorys_table[id][w_rank].recv_flag[0])
                            break;
                    }
            }
            } else {
            // All_reduce path
            auto start_3 = perf_dump_start();
            auto dst_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(0));
            auto dst_cl_buf = dst_mem->get_buffer().get();
            auto& adder_instance = get_adder_instance(w_rank);
            // auto data_size = dst_mem->size();
            for (size_t idx = 0; idx < w_size; idx++) {
                if (idx != static_cast<size_t>(w_rank)) {
                    auto src_mem = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[idx];
                    auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem)->get_buffer().get();
                    sync_event = adder_instance.tensor_add(
                        stream,
                        src_cl_buf,
                        dst_cl_buf,
                        dst_mem->count(),
                        adder_instance.element_type_to_kernel_data_type(dst_mem->get_layout().data_type));
                    // sync_event->wait();
                    sync_events.emplace_back(sync_event);
                }
            }
            const auto end_add = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> elapsed_1 = end_add - start_3;
            if (0) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                std::cout << "Add Solution[" << all_reduce_solution << "] " << "Rank[" << w_rank
                          << "] sync tensor p2p add total cost: " << elapsed_1.count() << " ms" << std::endl;
            }
            perf_dump_done(start_3,
                           std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor allreduce add"),
                           true);
            }
        } else {
            auto src_mem = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[w_rank];
            auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem)->get_buffer().get();
            auto dst_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(0));
            auto dst_cl_buf = dst_mem->get_buffer().get();
            auto sync_event = gpu_p2p_instance.remote_copy_rect(stream,
                                                                src_cl_buf,
                                                                instance.get_output_layout(1),
                                                                dst_cl_buf,
                                                                instance.get_output_layout(0),
                                                                w_rank,
                                                                false);
            sync_events.emplace_back(sync_event);
        }
        }
        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }
        perf_dump_done(start, std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor total"), true);

        const auto end_xj = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end_xj - start;
        if (0) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << "Solution[" << all_reduce_solution << "] " << "Rank[" << w_rank
                      << "] sync tensor p2p add total cost: " << elapsed_1.count() << " ms" << std::endl;
        }

        // This block MUST be put exactly at the end of this method.
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_memorys_table[id][w_rank].layout = p2p_src_layout;
            sub_mem_mgr->_use_count[id]++;
        }
        // sub_mem_mgr->_memorys_table[id][w_rank].layer_finished_flag = true;
        // while (true) {
        //             size_t wait_all_ouput_ready = 0;
        //             for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
        //                 if (sub_mem_mgr->_memorys_table[id][w_rank].layer_finished_flag == true)
        //                     wait_all_ouput_ready++;
        //             }
        //             if (wait_all_ouput_ready == w_size)
        //                 break;
        //         }
        return sync_events.size() > 0 ? stream.group_events(sync_events) : stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<sync_tensor_impl>();
    }

    std::vector<void*> all_gather_remote_dst;
    std::vector<void*> all_reduce_remote_dst;
    cl_mem all_gather_current_dst;
};

namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
