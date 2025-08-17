#!/bin/bash

set -ex
rm -Rf build
mkdir build
pushd build
export VERBOSE=1
# cmake -DCMAKE_TOOLCHAIN_FILE=../clang-15.cmake ..
ccmake ..
make -j 8
./test

