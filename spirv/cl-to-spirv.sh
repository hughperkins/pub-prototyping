#!/bin/bash
# designed to be run fomr subidrecotyr build

# source ~/git/marsohodmethod/activate
# source ~/spirv-tools/activate
export PATH=~/git-local/spirv/dist/bin:$PATH
export PATH=~/git-local/spirv/spirv-tools/bin:$PATH

# clang -cc1 -emit-llvm-bc -triple spir-unknown-unknown '' cl-spir-compile-options "" -include opencl_spir.h -o cl_kernel1.spv cl_kernel1.cl
clang -cc1 -emit-spirv -triple spir-unknown-unknown -cl-std=CL1.2 -include opencl.h -x cl -o cl_kernel1.spv ../cl_kernel1.cl
spirv-dis cl_kernel1.spv -o cl_kernel1.ll

