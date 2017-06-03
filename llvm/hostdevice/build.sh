#!/bin/bash

set -x
set -e

mkdir -p build

clang++-3.8 -x cuda -std=c++11 --cuda-host-only -emit-llvm  -O0 -S somecode.cu -o somecode-hostraw.ll
clang++-3.8 -x cuda -std=c++11 --cuda-device-only -emit-llvm  -O3 -S somecode.cu -o somecode-device.ll

grep define somecode-device.ll 
grep define somecode-hostraw.ll 
