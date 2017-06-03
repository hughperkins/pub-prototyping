#!/bin/bash

set -x
set -e

# COCL_HOME=~/git/cuda-on-cl   # or wherever it is

mkdir -p build
clang++-3.8 -DUSE_CLEW -I/usr/local/include/EasyCL -fPIC -c -g -o build/cocl-issue3-d.o cocl-issue3-d.cpp -std=c++11
g++ -fPIC -pie -Wl,-rpath,/usr/local/lib -g -o build/cocl-issue3-d build/cocl-issue3-d.o -leasycl -lclew -lcocl -lclblast -lpthread
build/cocl-issue3-d
