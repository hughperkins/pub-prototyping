cmake_minimum_required(VERSION 2.8)

set(LIBDNN_INSTALL_DIR CACHE PATH "directory you installed libdnn to")
set(VIENNACL_DIR CACHE PATH "viennacl dir")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(test1 test1.cpp)
target_link_libraries(test1 OpenCL)
target_link_libraries(test1 ${LIBDNN_INSTALL_DIR}/lib/libgreentea_libdnn.so)
include_directories(${LIBDNN_INSTALL_DIR}/include)
include_directories(${VIENNACL_DIR})
