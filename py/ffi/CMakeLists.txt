cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -g -Wno-sign-compare")

add_library(mylib SHARED myfunc.cpp)
