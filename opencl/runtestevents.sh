#!/bin/bash

set -x
set -e

mkdir -p build
g++ -std=c++11 -o build/testevents testevents.cpp -lOpenCL
build/testevents
