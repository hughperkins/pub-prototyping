// #include "tensorflow/core/common_runtime/gpu/gpu_device.h"
// #include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/common_runtime/device_factory.h"
// #include "tensorflow/core/common_runtime/threadpool_device.h"

#include <iostream>

namespace tensorflow {

// class MyGPUDevice : public BaseGPUDevice {
//  public:
//   MyGPUDevice(const SessionOptions& options, const string& name,
//             Bytes memory_limit, BusAdjacency bus_adjacency, int gpu_id,
//             const string& physical_device_desc, Allocator* gpu_allocator,
//             Allocator* cpu_allocator)
//       : BaseGPUDevice(options, name, memory_limit, bus_adjacency, gpu_id,
//                       physical_device_desc, gpu_allocator, cpu_allocator,
//                       false /* sync every op */, 1 /* max_streams */) {}

//   Allocator* GetAllocator(AllocatorAttributes attr) override {
//     std::cout << "GpuDevice::GetAllocator" << std::endl;
//     return 0;
//   }
// };

class MyGPUDeviceFactory : public DeviceFactory {
public:
  MyGPUDeviceFactory() {
    std::cout << "MyGPUDeviceFactory" << std::endl;
  }
  virtual tensorflow::Status CreateDevices(const tensorflow::SessionOptions&, const string&, std::vector<tensorflow::Device*>*) {
    std::cout << "MyGPUDeviceFacotry::CreateDevices" << std::endl;
    return Status::OK();
  }
};

// REGISTER_LOCAL_DEVICE_FACTORY("GPU", MyGPUDeviceFactory);

}

extern "C" {
  void fact_init();
}

using namespace tensorflow;

void fact_init() {
  REGISTER_LOCAL_DEVICE_FACTORY("GPU", MyGPUDeviceFactory);
}

