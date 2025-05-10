#!/bin/bash

set -ex
rm -Rf build
mkdir build
pushd build
# ccmake -DCMAKE_TOOLCHAIN_FILE=../clang-15.cmake ..
ccmake ..

