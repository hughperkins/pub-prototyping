cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS -std=c++11)
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include("${CMAKE_MODULE_PATH}/FindOpenCL.cmake")

add_executable(testmapbuffer testmapbuffer.cpp)
target_link_libraries(testmapbuffer easycl clew dl)
target_compile_definitions(testmapbuffer PRIVATE -DUSE_CLEW)
set_target_properties(testmapbuffer PROPERTIES
     LINK_FLAGS -Wl,-rpath,/usr/local/lib)

add_executable(test_buffers test_buffers.cpp)
target_link_libraries(test_buffers easycl dl)
target_link_libraries(test_buffers ${OPENCL_LIBRARIES})
