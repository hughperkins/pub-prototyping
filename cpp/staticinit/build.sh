#!/bin/bash

# clang++-3.8 -c m1.cpp -emit-ll -S -o m1.ll

set -e
set -x

mkdir -p build

clang++-3.8 -std=c++11 -c m1.cpp -o build/m1.o
clang++-3.8 -std=c++11 -c s1.cpp -o build/s1.o
# clang++-3.8 -std=c++11 -c s2.cpp -o build/s2.o
# clang++-3.8 -S -emit-llvm -std=c++11 -c s2.cpp -o build/s2.ll
clang++-3.8 -c -o build/s2.o s2.ll
clang++-3.8 -std=c++11 -c s3.cpp -o build/s3.o
g++ -o build/test build/m1.o build/s1.o build/s2.o build/s3.o
build/test
