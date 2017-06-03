#!/bin/bash

THISDIR=$(pwd)

# cd ~/git/tensorflow-blas
# g++ -std=c++11 -fPIC -c \
#     -I. \
#     -I${THISDIR} \
#     -I$HOME/git/eigen \
#      -o /tmp/foo \
#      tensorflow/core/kernels/sparse_tensor_dense_matmul_op.cc

g++ -std=c++11 -fPIC \
     -DEIGEN_MPL2_ONLY -DEIGEN_AVOID_STL_ARRAY\
     -I$HOME/git/eigen/unsupported \
     -I$HOME/git/eigen \
     -c test.cpp -o /tmp/foo.o
