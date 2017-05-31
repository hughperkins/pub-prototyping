#!/bin/bash

set -e

mkdir -p build
clang++-3.8 -std=c++11 test_uniqueptr.cpp -O3 -o build/test_uniqueptr_clang
build/test_uniqueptr_clang
