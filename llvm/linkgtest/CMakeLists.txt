cmake_minimum_required(VERSION 2.8)

set(CLANG_HOME "/usr/lib/llvm-3.8" CACHE STRING "eg the downloaded clang-3.8.0 folder, containing lib, bin etc")

set(CMAKE_CXX_FLAGS -std=c++11)

# execute_process(
#     # COMMAND "${CLANG_HOME}/bin/llvm-config --ldflags --libs | tr -d '\n'"
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-llvm-link-config.sh ${CLANG_HOME}
#     OUTPUT_VARIABLE LLVM_LDFLAGS
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# message("LLVM_LDFLAGS ${LLVM_LDFLAGS}")

# execute_process(
#     # COMMAND "${CLANG_HOME}/bin/llvm-config --ldflags --libs | tr -d '\n'"
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-llvm-cxxflags.sh ${CLANG_HOME}
#     OUTPUT_VARIABLE LLVM_CXXFLAGS
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# message("LLVM_CXXFLAGS ${LLVM_CXXFLAGS}")

execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm-syslibs.sh ${CLANG_HOME}
    OUTPUT_VARIABLE LLVM_SYSLIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get-llvm-libs.sh ${CLANG_HOME}
    OUTPUT_VARIABLE LLVM_LIBFILES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

STRING(REGEX MATCHALL "[^ ]+" LLVM_LIBFILES "${LLVM_LIBFILES}")

# LLVM_LIBS
message("LLVM_LIBFILES ${LLVM_LIBFILES}")
foreach(LIB ${LLVM_LIBFILES})
   message("LIB ${LIB}")
   if(LLVM_LIBS)
   set(LLVM_LIBS ${LLVM_LIBS} ${CLANG_HOME}/lib/${LIB})
   else()
   set(LLVM_LIBS "${CLANG_HOME}/lib/${LIB}")
   endif()
endforeach()
message("LLVM_LIBS [${LLVM_LIBS}]")
message("LLVM_LIBFILES ${LLVM_LIBFILES}")

# set(LLVM_LDFLAGS "-L/usr/lib/llvm-3.8/lib -lLLVMCore -lLLVMSupport -lrt -ldl -ltinfo -lpthread -lz -lm")

add_executable(mytest mytest.cpp gtest_main.cpp GtestGlobals.cpp EasyCL/util/easycl_stringhelper.cpp gtest-all.cc)
target_include_directories(mytest PRIVATE ${CLANG_HOME}/include)
target_include_directories(mytest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/EasyCL)
target_link_libraries(mytest ${LLVM_LIBS} ${LLVM_SYSLIBS})
# get_target_property(mytest
#     PROPERTY EXISTINGLINK_FLAGS
# )
# message("EXISTINGLINK_FLAGS ${EXISTINGLINK_FLAGS}")
# set_target_properties(mytest PROPERTIES
#     LINK_FLAGS "-L{CLANG_HOME}/lib"
# )
