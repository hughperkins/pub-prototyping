cmake_minimum_required(VERSION 3.5)

set(TFDIR "CHANGEME" CACHE STRING "cloned tensorflow repo")
set(EIGENDIR "CHANGEME" CACHE STRING "cloned eigen repo")

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -g")

add_executable(testphilox testphilox.cpp ../third_party/argparsecpp/argparsecpp.cpp)
target_include_directories(testphilox PRIVATE ${TFDIR} ${EIGENDIR})

add_executable(test_cuda_device_array test_cuda_device_array.cpp)
target_include_directories(test_cuda_device_array PRIVATE ${TFDIR} ${EIGENDIR})
