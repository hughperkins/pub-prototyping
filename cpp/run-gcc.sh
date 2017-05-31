#!/bin/bash

set -e

mkdir -p build
g++ -std=c++11 test_uniqueptr.cpp -O3 -o build/test_uniqueptr_gcc
build/test_uniqueptr_gcc
