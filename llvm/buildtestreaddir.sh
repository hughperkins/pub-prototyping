#!/bin/bash

LLVM_CONFIG=llvm-config-3.8
LINK_FLAGS=`${LLVM_CONFIG} --ldflags --system-libs --libs all`

g++ -std=c++11 -I/usr/lib/llvm-3.8/include -o /tmp/testreaddir /home/ubuntu/prototyping/testreadir.cpp ${LINK_FLAGS}
