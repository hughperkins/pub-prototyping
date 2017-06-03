cmake_minimum_required(VERSION 2.8)

include_directories(/home/ubuntu/git/clew/include/proxy-opencl)
include_directories(/home/ubuntu/torch/install/include)

add_executable(gemmtest gemmtest.cpp)
target_link_libraries(gemmtest dl)
target_link_libraries(gemmtest /home/ubuntu/torch/install/lib/libclBLAS.so)
target_link_libraries(gemmtest /home/ubuntu/torch/install/lib/libclew.so)
target_link_libraries(gemmtest /usr/lib/libblas.so)
target_link_libraries(gemmtest libpthread.so)

