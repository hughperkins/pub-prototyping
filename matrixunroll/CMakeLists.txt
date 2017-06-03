cmake_minimum_required(VERSION 2.8)

set( CMAKE_CXX_FLAGS "-std=c++0x")

#find_package(OpenCL REQUIRED)

add_executable(convrolled convrolled.cpp stringhelper.cpp )
add_executable(convunrolled convunrolled.cpp stringhelper.cpp unrolling_cpu.cpp )
add_executable(unrollclblas unrollclblas.cpp stringhelper.cpp unrolling_cpu.cpp )
add_executable(rolledprop rolledprop.cpp stringhelper.cpp )
add_executable(testclblas testclblas.cpp clew.c)
add_executable(testclblas2 testclblas2.cpp clew.c)

target_link_libraries( testclblas dl )
target_link_libraries( testclblas2 dl )

target_include_directories( unrollclblas PRIVATE ~/git/clBLAS/src)
target_include_directories( unrollclblas PRIVATE ~/git/OpenCLHelper)
target_include_directories( testclblas PRIVATE ~/git/clBLAS/src)
target_include_directories( testclblas2 PRIVATE ~/git/clBLAS/src)
target_include_directories( testclblas2 PRIVATE ~/git/OpenCLHelper)
target_include_directories( rolledprop PRIVATE ~/git/OpenCLHelper)

target_link_libraries( unrollclblas ~/git/clBLAS/src/build/library/libclBLAS.so)
target_link_libraries( unrollclblas ~/git/OpenCLHelper/build/libOpenCLHelper.so)
target_link_libraries( rolledprop ~/git/OpenCLHelper/build/libOpenCLHelper.so)
target_link_libraries( testclblas ~/git/clBLAS/src/build/library/libclBLAS.so)
target_link_libraries( testclblas2 ~/git/clBLAS/src/build/library/libclBLAS.so)
target_link_libraries( testclblas2 ~/git/OpenCLHelper/build/libOpenCLHelper.so    )

