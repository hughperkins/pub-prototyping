cmake_minimum_required(VERSION 2.8)

include_directories(/usr/lib/jvm/java-7-openjdk-amd64/include/)

include_directories(/data/install/svmlight)

add_library(jnitest SHARED jnitest.cpp )
add_library(jnitest_c SHARED jnitest_c.c jnitest_c_wrap.c )

add_executable(testmyfunc testmyfunc.cpp)
target_link_libraries(testmyfunc jnitest)

add_executable(testmyfunc_c testmyfunc_c.c)
target_link_libraries(testmyfunc_c jnitest_c)

