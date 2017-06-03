#include <iostream>
#include <functional>

typedef uint32_t uint32;
typedef int32_t int32;
typedef uint64_t uint64;
typedef int64_t int64;
typedef uint16_t uint16;
typedef int8_t int8;

// #define DT_INT8 int8_t
template<typename T>
class DT {
public:
    static const DT<T> get() {
        const DT<T> v;
        return v;
    }
};
#define DT_INT8 DT<int8_t>::get()

#define EIGEN_DEVICE_FUNC __attribute__((device))
#define EIGEN_STRONG_INLINE __inline__

#define TF_RETURN_IF_ERROR(x) x

class Status {
public:
    Status(int value) {
        this->value = value;
    }
    static Status OK() {
        return Status(0);
    }
    int value;
};

struct TensorShape {
    int64_t x;
};

class AllocatorAttributes {
public:
    void set_on_host(bool v) {
        this->on_host = v;
    }
    void set_gpu_compatible(bool v) {
        this->gpu_compatible = v;
    }
    bool on_host = false;
    bool gpu_compatible;
};

namespace perftools {
    namespace gputools {
        struct DeviceMemoryBase {
            float *mem;
            uint64_t size;
        };
    }
}

class Tensor {
public:
    template<typename T>
    Tensor flat() {
        std::cout << "Tensor::flat()" << std::endl;
        Tensor ret;
        return ret;
    }
    bool IsInitialized() const;
    float *data();
};

class TensorReference {
public:
    TensorReference(Tensor tensor) {
        std::cout << "TensorReference::TensorReference()" << std::endl;
    }
    void Unref() const;
};

class Stream;
class Device;
class GpuDeviceInfo;

class OpDeviceContext {
public:
    OpDeviceContext();
    ~OpDeviceContext();
    Stream *stream();
    Stream *p_stream;
};

class Device {
public:
    Device();
    ~Device();
    GpuDeviceInfo *tensorflow_gpu_device_info();
    GpuDeviceInfo *p_gpuDeviceInfo;
};

class OpKernelContext {
public:
    Status allocate_temp(DT<int8_t> dt, TensorShape tensorShape, Tensor *tensor, AllocatorAttributes attrib) {
        std::cout << "OpKernelContext::allocate_temp(dt, shape, tensor, attrib)" << std::endl;
        return Status::OK();
    }
    Status allocate_temp(DT<int8_t> dt, TensorShape tensorShape, Tensor *tensor) {
        std::cout << "OpKernelContext::allocate_temp(dt, shape, tensor)" << std::endl;
        return Status::OK();
    }
    OpDeviceContext *op_device_context() {
        return &opDeviceContext;
    }
    Device *device() {
        std::cout << "OpKernelContext::device()" << std::endl;
        return &_device;
    }
    OpDeviceContext opDeviceContext;
    Device _device;
};

class Stream {
public:
    void ThenMemcpy(perftools::gputools::DeviceMemoryBase *, float *, uint32_t total_bytes);
};
    // stream->ThenMemcpy(&output_values_base,
    //                    out_of_line_values_on_host_.flat<int8>().data(),
    //                    total_bytes_);

class EventMgr {
public:
    void ThenExecute(Stream *stream, std::function<void ()> fn);
};

class GpuDeviceInfo {
public:
    GpuDeviceInfo() {
        event_mgr = new EventMgr();
    }
    ~GpuDeviceInfo() {
        delete event_mgr;
    }
    EventMgr *event_mgr;
};

OpDeviceContext::OpDeviceContext() {
    p_stream = new Stream();
}
OpDeviceContext::~OpDeviceContext() {
    delete p_stream;
    p_stream = 0;
}
Stream *OpDeviceContext::stream() {
    std::cout << "OpDeviceContext::stream()" << std::endl;
    return p_stream;
}

Device::Device() {
    p_gpuDeviceInfo = new GpuDeviceInfo();
}
Device::~Device() {
    delete p_gpuDeviceInfo;
}
GpuDeviceInfo *Device::tensorflow_gpu_device_info() {
    std::cout << "Device::GpuDeviceInfo()" << std::endl;
    return p_gpuDeviceInfo;
}

void EventMgr::ThenExecute(Stream *stream, std::function<void ()> fn) {
    std::cout << "EventMgr::ThenExecute(stream, fn)" << std::endl;
}

void Stream::ThenMemcpy(perftools::gputools::DeviceMemoryBase *, float *, uint32_t total_bytes) {
    std::cout << "Stream::ThenMemcpy(memorybase, float *, bytes=" << total_bytes << std::endl;
}

float *Tensor::data() {
    std::cout << "Tensor::data()" << std::endl;
    return 0;
}

void TensorReference::Unref() const {
    std::cout << "TensorReference::Unref()" << std::endl;
}

#define TF_DISALLOW_COPY_AND_ASSIGN(x)

#include "tf_files/cuda_device_array_gpu.h"
#include "tf_files/cuda_device_array.h"

int main(int argc, char *argv[]) {
    // based loosely on tensorflow split_op.cc:
    OpKernelContext opKernelContext;
    int num_split = 4;
    tensorflow::CudaDeviceArrayOnHost<float *> ptrs(&opKernelContext, num_split);
    ptrs.Init();
    ptrs.Finalize();
    return 0;
}

    // for (int i = 0; i < num_split; ++i) {
    //   Tensor* result = nullptr;
    //   OP_REQUIRES_OK(context,
    //                  context->allocate_output(i, output_shape, &result));
    //   std::cout << "  SplitOpGPU::Compute() i=" << i << " result=" << result << " num_split=" << num_split << std::endl;
    //   std::cout << "  SplitOpGPU::Compute() result->flat<T>().data() " << result->flat<T>().data() << std::endl;
    //   ptrs.Set(i, result->flat<T>().data());
    // }
    // if (prefix_dim_size * split_dim_output_size * suffix_dim_size == 0) {
    //   return;
    // }
    // OP_REQUIRES_OK(context, ptrs.Finalize());

    // std::cout << "  SplitOpGPU::Compute() call .Run(eigen_device, data=" << input.flat<T>().data() <<
    //     " prefix_dim_size=" << prefix_dim_size << " split_dim_size=" << split_dim_size << " ptrs.data().size=" << ptrs.data().size << ")" << std::endl;
    // SplitOpGPULaunch<T>().Run(context->eigen_device<GPUDevice>(),
    //                           input.flat<T>().data(), prefix_dim_size,
    //                           split_dim_size, suffix_dim_size, ptrs.data());
