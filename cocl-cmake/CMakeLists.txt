cmake_minimum_required(VERSION 2.8)

set(EASYCL_SRCS src/EasyCL/CLKernel.cpp src/EasyCL/CLWrapper.cpp src/EasyCL/platforminfo_helper.cpp
    src/EasyCL/deviceinfo_helper.cpp src/EasyCL/util/easycl_stringhelper.cpp src/EasyCL/DevicesInfo.cpp src/EasyCL/DeviceInfo.cpp)

set(COCL_SRCS src/hostside_opencl_funcs.cpp src/cocl_events.cpp src/cocl_blas.cpp src/cocl_device.cpp src/cocl_error.cpp
    src/cocl_memory.cpp src/cocl_properties.cpp src/cocl_streams.cpp src/cocl_clsources.cpp src/cocl_context.cpp)

include_directories(src/EasyCL/thirdparty/clew/include/proxy-opencl)
include_directories(src/EasyCL/thirdparty/clew/include)
include_directories(include)
include_directories(src/EasyCL)
include_directories(src/CLBlast/include)
include_directories(/usr/include/llvm-3.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")
add_definitions(-DUSE_CLEW)
# for llvm:
add_definitions(-D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-rpath,$ORIGIN")

add_executable(ir-to-opencl
    src/ir-to-opencl.cpp src/ir-to-opencl-common.cpp src/mutations.cpp src/readIR.cpp src/struct_clone.cpp
    src/handle_branching.cpp src/branching_transforms.cpp src/flowcontrolinstructions.cpp)
target_include_directories(ir-to-opencl PRIVATE src)

add_library(clew SHARED src/EasyCL/thirdparty/clew/src/clew.c)
add_library(easycl SHARED ${EASYCL_SRCS})
add_library(cocl SHARED ${COCL_SRCS})

# SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
# FindPackage(cocl)
include(cmake/cocl.cmake)

# ==================================================================================================

# next ~20 lines or so are copied from CLBlast CMakeLists.txt (seems easier than figuring out the whole cmake import/export
# bit...)
# I prefixed the vars with CLBLAST_ , and removed all the level 1/2 routines, which we dont use

# Sets the supported routines and the used kernels. New routines and kernels should be added here.
set(CLBLAST_KERNELS copy_fast copy_pad transpose_fast transpose_pad xaxpy xdot xger xgemm xgemv)
# set(LEVEL1_ROUTINES xswap xscal xcopy xaxpy xdot xdotu xdotc xnrm2 xasum xamax)
# set(LEVEL2_ROUTINES xgemv xgbmv xhemv xhbmv xhpmv xsymv xsbmv xspmv xtrmv xtbmv xtpmv
#                     xger xgeru xgerc xher xhpr xher2 xhpr2 xsyr xspr xsyr2 xspr2)
set(CLBLAST_LEVEL3_ROUTINES xgemm xsymm xhemm xsyrk xherk xsyr2k xher2k xtrmm)
set(CLBLAST_LEVELX_ROUTINES xomatcopy)
set(CLBLAST_ROUTINES ${CLBLAST_LEVEL3_ROUTINES} ${CLBLAST_LEVELX_ROUTINES})
set(CLBLAST_PRECISIONS 32 64 3232 6464 16)

# Gathers all source-files
set(CLBLAST_SOURCES
  src/CLBlast/src/database/database.cpp
  src/CLBlast/src/routines/common.cpp
  src/CLBlast/src/cache.cpp
  src/CLBlast/src/clblast.cpp
  src/CLBlast/src/clblast_c.cpp
  src/CLBlast/src/routine.cpp
  src/CLBlast/src/utilities.cpp
)
# foreach(ROUTINE ${LEVEL1_ROUTINES})
#   set(CLBLAST_SOURCES ${CLBLAST_SOURCES} src/routines/level1/${ROUTINE}.cpp)
# endforeach()
# foreach(ROUTINE ${LEVEL2_ROUTINES})
#   set(CLBLAST_SOURCES ${CLBLAST_SOURCES} src/routines/level2/${ROUTINE}.cpp)
# endforeach()
foreach(ROUTINE ${LEVEL3_ROUTINES})
  set(CLBLAST_SOURCES ${CLBLAST_SOURCES} src/CLBlast/src/routines/level3/${ROUTINE}.cpp)
endforeach()
foreach(ROUTINE ${LEVELX_ROUTINES})
  set(CLBLAST_SOURCES ${CLBLAST_SOURCES} src/CLBlast/src/routines/levelx/${ROUTINE}.cpp)
endforeach()

include_directories(src/CLBlast/src)
add_library(clblast SHARED ${CLBLAST_SOURCES})

# ==================================================================================================
# tests

add_cocl_executable(testblas test/cocl/testblas.cu)
