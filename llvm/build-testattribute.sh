#!/bin/bash

set -e
set -x

mkdir -p build
clang++-3.8 -c -o build/testattribute.ll -emit-llvm -S testattribute.cpp
clang++-3.8 -c -o build/testattribute.o build/testattribute.ll
g++ -o build/testattribute build/testattribute.o
