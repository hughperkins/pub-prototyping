cmake_minimum_required(VERSION 2.8)

include_directories(/usr/include/mpich2)
add_library( mpiwrapper SHARED mpiwrapper.c )

add_executable( testwrapper testwrapper.cpp )
target_link_libraries( testwrapper mpiwrapper mpich mpl )

