cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(BOOST_HOME "" CACHE STRING "the untarred boost_1_64_0.tar.bz2 folder")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
add_executable(try_boost_graph_library try_boost_graph_library.cpp)
