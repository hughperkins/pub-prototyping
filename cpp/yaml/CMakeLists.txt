cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_FLAGS -std=c++11)

add_executable(test_yaml test_yaml.cpp)
target_link_libraries(test_yaml $ENV{HOME}/git-local/yaml-cpp/build/libyaml-cpp.dylib)
target_include_directories(test_yaml PRIVATE ~/git-local/yaml-cpp/include)
