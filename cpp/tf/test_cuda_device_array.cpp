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
class TensorReference {
public:
    void Unref() const;
};
namespace perftools {
    namespace gputools {
        class DeviceMemoryBase {
        public:
        };
    }
}
class Stream {
public:
    void ThenMemcpy(float *);
};
class EventMgr {
public:
    void ThenExecute(Stream &stream, std::function<void ()> fn);
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
class Tensor {
public:
    template<typename T>
    Tensor flat() {
        std::cout << "Tensor::flat()" << std::endl;
        Tensor ret;
        return ret;
    }
    bool IsInitialized() const;
    float *data() {
        std::cout << "Tensor::data()" << std::endl;
        return 0;
    }
};
class Device {
public:
    GpuDeviceInfo *tensorflow_gpu_device_info();
};
class OpDeviceContext {
public:
    Stream *stream();
};
class OpKernelContext {
public:
    Status allocate_temp(DT<int8_t> dt, TensorShape tensorShape, Tensor *tensor, AllocatorAttributes attrib) {
        std::cout << "OpKernelContext::allocate_temp()" << std::endl;
        return Status::OK();
    }
    OpDeviceContext *op_device_context() {
        return &opDeviceContext;
    }
    Device *device();
    OpDeviceContext opDeviceContext;
};
#define TF_DISALLOW_COPY_AND_ASSIGN(x)

#include "tf_files/cuda_device_array_gpu.h"
#include "tf_files/cuda_device_array.h"

int main(int argc, char *argv[]) {
    OpKernelContext opKernelContext;
    int num_split = 4;
    tensorflow::CudaDeviceArrayOnHost<float *> ptrs(&opKernelContext, num_split);
    ptrs.Init();
    return 0;
}
