cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

if(WIN32)
  set(CMAKE_CXX_FLAGS "/EHsc")
else()
  SET(PLATFORM_OPTIONS "")
  if(UNIX)
      SET(PLATFORM_OPTIONS "-Wno-sign-compare")
  endif()
  if(APPLE)
    set(PLATFORM_OPTIONS "${PLATFORM_OPTIONS} -Wno-covered-switch-default")
  endif()

  set(CMAKE_CC_FLAGS "-fPIC")
  set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -g ${PLATFORM_OPTIONS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN")
endif()

set(SPIRV_HOME "/usr/local/opt/spirv" CACHE PATH "directory containing the llvm/clang/llvm installed files")
set(SPIRVTOOLS_HOME "/usr/local/opt/spirv-tools" CACHE PATH "directory containing the sirv-tools installed files")

execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-llvm-cxxflags.sh ${SPIRV_HOME}
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
STRING(REGEX REPLACE " " ";" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
string(REGEX REPLACE ";-std=c\\+\\+0x" ";-std=c++11" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
string(REGEX REPLACE ";-fno-exceptions" ";-fexceptions" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
string(REGEX REPLACE ";-DNDEBUG" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
string(REGEX REPLACE ";-isysroot;[^;]" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

# split LLVM_CXXFLAGS into defines vs other options
STRING(REGEX MATCHALL ";-D[^;]+" LLVM_DEFINES "${LLVM_CXXFLAGS}")
STRING(REGEX REPLACE ";+" ";" LLVM_DEFINES "${LLVM_DEFINES}")
STRING(REGEX REPLACE ";-D[^;]+" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

if(WIN32)
    execute_process(
        COMMAND ${SPIRV_HOME}\\bin\\llvm-config --libnames
        OUTPUT_VARIABLE LLVM_LIBNAMES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND ${SPIRV_HOME}\\bin\\llvm-config --system-libs
        OUTPUT_VARIABLE LLVM_SYSLIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    # syslibs is non-llvm libs, used by llvm, eg -lz -lm -curses
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm-syslibs.sh ${SPIRV_HOME}
        OUTPUT_VARIABLE LLVM_SYSLIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # llvm-libs is the actual llvm libs, like LLVMCore, etc
    execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-llvm-libs.sh ${SPIRV_HOME}
        OUTPUT_VARIABLE LLVM_LIBNAMES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
separate_arguments(LLVM_LIBNAMES)
separate_arguments(LLVM_SYSLIBS)
# at this point, LLVM_LIBNAMES should be semi-colon separated, comprising llvm libfile names, without
# paths. We will add the path now:
# output of this bit is that LLVM_LIBPATHS should be semi-colon separated, and each item
# should be a full filepath of one llvm link library
foreach(LIBNAME ${LLVM_LIBNAMES})
    set(LLVM_LIBPATHS ${LLVM_LIBPATHS} ${SPIRV_HOME}/lib/${LIBNAME})
endforeach()
separate_arguments(PLATFORM_OPTIONS)

add_executable(test_spir_cpp test_spir_cpp.cpp)
# target_include_directories(test_spir_cpp PRIVATE "${SPIRV_HOME}/include")
target_include_directories(test_spir_cpp PRIVATE ${SPIRVTOOLS_HOME}/include)
# target_compile_options(test_spir_cpp PRIVATE ${LLVM_CXXFLAGS} ${LLVM_DEFINES} ${PLATFORM_OPTIONS})
# target_compile_options(test_spir_cpp PRIVATE ${PLATFORM_OPTIONS})
# target_link_libraries(test_spir_cpp ${LLVM_LIBPATHS} ${LLVM_SYSLIBS} ${SPIRVTOOLS_HOME}/lib/libSPIRV-Tools.a)
target_link_libraries(test_spir_cpp ${SPIRVTOOLS_HOME}/lib/libSPIRV-Tools.a)
