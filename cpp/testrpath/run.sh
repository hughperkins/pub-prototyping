#!/bin/bash

set -x
set -e

mkdir -p build
g++ -c -fPIC -o build/foo.o foo.cpp
g++ -c -fPIC -o build/bar.o bar.cpp
g++ -c -fPIC -o build/main.o main.cpp

mkdir -p build/somedir
g++ -shared -o build/somedir/libbar.so build/bar.o 
# -Wl,-rpath=\$ORIGIN/somedir 
g++ -shared -o build/libfoo.so build/foo.o  -Wl,-rpath=\$ORIGIN/somedir -Lbuild/somedir -lbar
# g++ -o build/main build/main.o -Lbuild -lfoo -Wl,-rpath=\$ORIGIN -Lbuild/somedir -lbar
g++ -o build/main build/main.o -Lbuild -lfoo -Lbuild/somedir -lbar

LD_LIBRARY_PATH=build build/main
