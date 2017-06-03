#!/bin/bash

set -e
set -x

mkdir -p build
clang++-3.8 -std=c++11 -g -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -I/usr/lib/llvm-3.8/include patchll.cpp -o build/patchll -L/usr/lib/llvm-3.8/lib -lLLVM

clang++-3.8 -emit-llvm -S -o build/test.ll test.cpp
build/patchll
clang++-3.8 -c -o build/test.o build/test-patched.ll
g++ -o build/test build/test.o
build/test
