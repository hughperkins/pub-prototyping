#!/bin/bash

g++ -g -I/usr/include -std=c++11 -I$HOME/git/EasyCL -I/norep/Downloads/boost_1_61_0 -o test1b test1b.cpp -lOpenCL -L$HOME/git/EasyCL/build-bc -lEasyCL || exit 1
./test1b
