#!/bin/bash

set -e

export CLANG=clang++-3.8
export LLVM_CONFIG=llvm-config-3.8
export LLVM_INCLUDE=/usr/include/llvm-3.8

# export COMPILE_FLAGS="$(${LLVM_CONFIG} --cxxflags) -std=c++11"
export LINK_FLAGS="$(${LLVM_CONFIG} --ldflags --system-libs --libs all)"
echo ${COMPILE_FLAGS}

COMPILE_FLAGS="-I/usr/lib/llvm-3.8/include -std=c++0x -fPIC -fvisibility-inlines-hidden -std=c++11 -ffunction-sections -fdata-sections -O2 -g -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -std=c++11"

clang++-3.8 ${COMPILE_FLAGS} -g -I/usr/include/llvm-3.8 testassert.cpp -o build/testassert ${LINK_FLAGS}
build/testassert "$@"
