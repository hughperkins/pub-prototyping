cmake_minimum_required(VERSION 2.8)

#find_library(LUA5.1 REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 ")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")

include_directories(/home/ubuntu/torch/install/include)
add_executable(teststack teststack.cpp)
target_link_libraries(teststack /home/ubuntu/torch/install/lib/libluajit.so dl)
target_link_libraries(teststack dl)

