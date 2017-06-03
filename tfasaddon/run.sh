#!/bin/bash

THISDIR=$(pwd)

# if [[ ! -f build/fact.o ]]; then {
    g++ -std=c++11 -fPIC \
        -I$HOME/git/tensorflow-blas \
        -I$HOME/git/eigen \
        -I/norep/envs/env3-base-nocuda/lib/python3.5/site-packages/tensorflow/include \
         -DEIGEN_MPL2_ONLY -DEIGEN_AVOID_STL_ARRAY\
         -c fact.cpp -o build/fact.o
# } fi

    # -I$HOME/prototyping/trytfbuild \
    # -I/norep/envs/env3-base-nocuda/lib/python3.5/site-packages/tensorflow/include \

g++ -o build/libfact.so -shared build/fact.o
