#!/bin/bash

set -x
set -e

# COCL_HOME=~/git/cuda-on-cl   # or wherever it is

mkdir -p build
clang++-3.8 -DUSE_CLEW -I/usr/local/include/EasyCL -fPIC -c -g -o build/cocl-issue3-f.o cocl-issue3-f.cpp -std=c++11
g++ -fPIC -pie -Wl,-rpath,/usr/local/lib -g -o build/cocl-issue3-f build/cocl-issue3-f.o -leasycl -lclew -lcocl -lclblast -lpthread
build/cocl-issue3-f
