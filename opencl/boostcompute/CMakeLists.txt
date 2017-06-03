cmake_minimum_required(VERSION 2.8)

option(USE_EASYCL "use easycl?" OFF)

if(USE_EASYCL)
add_definitions(-DUSE_EASYCL)
set(EASYCL_DIST /home/ubuntu/git/EasyCL/dist CACHE string "easycl dist directory" )
# set(EASYCL_LIB /home/ubuntu/git/EasyCL/build-bc CACHE stirng "directory containing easycl so" )
endif()

set(BOOST_INCLUDE "" CACHE string "directory containing boost dir" )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(USE_EASYCL)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS)
add_definitions(-DUSE_CLEW)
endif()
# add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_0_APIS)

add_executable(test1b test1b.cpp)
include_directories(${BOOST_INCLUDE})

if(USE_EASYCL)
include_directories(${EASYCL_DIST}/include)
include_directories(${EASYCL_DIST}/include/proxy-opencl)
include_directories(${EASYCL_DIST}/include/easycl)
target_link_libraries(test1b ${EASYCL_DIST}/lib/libEasyCL.so ${EASYCL_DIST}/lib/libclew.so)
else()
target_link_libraries(test1b OpenCL)
endif()
