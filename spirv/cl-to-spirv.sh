#!/bin/bash
# designed to be run fomr subidrecotyr build

# source ~/git/marsohodmethod/activate
# source ~/spirv-tools/activate
export PATH=~/git-local/spirv/dist/bin:$PATH
export PATH=~/git-local/spirv/spirv-tools/bin:$PATH


# .cl => .spv (SPIR-V binary)
clang -cc1 -emit-spirv -triple spir-unknown-unknown -cl-std=CL1.2 -include opencl.h -x cl -o cl_kernel1.spv ../cl_kernel1.cl

# .spv => .spt (SPIR-V binary => SPIR-V text)
spirv-dis cl_kernel1.spv -o cl_kernel1.spt

# .spv => .bc (SPIR-V binary => llvm binary)
llvm-spirv -r cl_kernel1.spv

# .bc => .ll (llvm binary => llvm text)
llvm-dis cl_kernel1.bc
