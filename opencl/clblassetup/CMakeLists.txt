cmake_minimum_required(VERSION 2.8)

include_directories(/home/ubuntu/git/clew/include/proxy-opencl)
include_directories(/home/ubuntu/torch/install/include)

add_executable(clblassetupteardown clblassetupteardown.cpp)
target_link_libraries(clblassetupteardown dl)
target_link_libraries(clblassetupteardown /home/ubuntu/torch/install/lib/libclBLAS.so)
target_link_libraries(clblassetupteardown /home/ubuntu/torch/install/lib/libclew.so)
target_link_libraries(clblassetupteardown /usr/lib/libblas.so)
target_link_libraries(clblassetupteardown libpthread.so)

add_executable(repeatgemm repeatgemm.cpp)
target_link_libraries(repeatgemm dl)
target_link_libraries(repeatgemm /home/ubuntu/torch/install/lib/libclBLAS.so)
target_link_libraries(repeatgemm /home/ubuntu/torch/install/lib/libclew.so)
target_link_libraries(repeatgemm /usr/lib/libblas.so)
target_link_libraries(repeatgemm libpthread.so)

