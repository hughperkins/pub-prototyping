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
    static DT<T> get() {
        DT<T> v;
        return v;
    }
};
#define DT_INT8 DT<int8_t>::get()

#define EIGEN_DEVICE_FUNC __attribute__((device))
#define EIGEN_STRONG_INLINE __inline__

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
class TensorShape {
public:
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
    Status allocate_temp();
    OpDeviceContext *op_device_context() {
        return &opDeviceContext;
    }
    Device *device();
    OpDeviceContext opDeviceContext;
};
class Tensor {
public:
    template<typename T>
    T *flat();
    bool IsInitialized() const;
};
#define TF_DISALLOW_COPY_AND_ASSIGN(x)
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

#include "tf_files/cuda_device_array_gpu.h"
#include "tf_files/cuda_device_array.h"

int main(int argc, char *argv[]) {
    OpKernelContext opKernelContext;
    int num_split = 4;
    tensorflow::CudaDeviceArrayOnHost<float *> ptrs(&opKernelContext, num_split);
    return 0;
}
