#!/bin/bash

set -x
set -e

mkdir -p build
touch build/foo
rm build/*

TARGET=h

# COCL_HOME=~/git/cuda-on-cl   # or wherever it is

cocl -fPIC -c cuda_sample_nomain.cu
clang++-3.8 -DUSE_CLEW -I/usr/local/include/EasyCL -fPIC -c -g -o build/${TARGET}.o ${TARGET}.cpp -std=c++11
g++ -fPIC -pie -Wl,-rpath,/usr/local/lib -g -o build/${TARGET} build/${TARGET}.o cuda_sample_nomain.o -leasycl -lclew -lcocl -lclblast -lpthread
build/${TARGET}
