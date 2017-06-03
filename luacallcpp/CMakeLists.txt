cmake_minimum_required(VERSION 2.8)

include_directories(/home/user/torch/install/include/TH)
add_library(foo SHARED foo.cpp)
add_executable(call_foo call_foo.cpp)
target_link_libraries(call_foo foo)
target_link_libraries(foo /home/user/torch/install/lib/libTH.so)

