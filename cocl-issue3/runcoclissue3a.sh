#!/bin/bash

set -x
set -e

# COCL_HOME=~/git/cuda-on-cl   # or wherever it is

mkdir -p build
clang++-3.8 -DUSE_CLEW -fPIC -c -o build/cocl-issue3-a.o cocl-issue3-a.cpp -std=c++11
g++ -fPIC -pie -Wl,-rpath,/usr/local/lib -o build/cocl-issue3-a build/cocl-issue3-a.o -leasycl -lclew
build/cocl-issue3-a
