#!/bin/bash

g++ -g -I/usr/include -std=c++11 -I/norep/Downloads/boost_1_61_0 -o test1 test1.cpp -lOpenCL || exit 1
./test1
