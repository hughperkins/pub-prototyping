#!/bin/bash

set -e

COMPILE_FLAGS="$(llvm-config-3.8 --cxxflags) -std=c++11"
LINK_FLAGS="$(llvm-config-3.8 --ldflags --system-libs --libs all)"

mkdir -p build
clang++-3.8 -emit-llvm caller.cpp -S -o caller.ll
clang++-3.8 -emit-llvm callee.cpp -S -o callee.ll

clang++-3.8 ${COMPILE_FLAGS} -fcxx-exceptions hackcaller.cpp -o build/hackcaller ${LINK_FLAGS}
build/hackcaller

# clang++-3.8 -c callee.ll -o build/callee.o
clang++-3.8 caller2.ll callee.ll -o build/caller
build/caller
