#!/bin/bash

set -x
set -e

CLANG_HOME=/usr/local/opt/llvm-4.0

# ${CLANG_HOME}/bin/clang++ -g -o simpleprog simpleprog.cpp

set +e
rm *.ll simpleprog *.dSYM *.o >/dev/null 2>&1
set -e

${CLANG_HOME}/bin/clang++ -std=c++11 -include /usr/local/include/cocl/cocl.h -x cuda \
    --cuda-host-only -nocudainc -nocudalib -g -S -emit-llvm -o simpleprog.ll simpleprog.cu

set +e
rm -R *.dSYM
set -e

${CLANG_HOME}/bin/clang++ -g -c -o simpleprog.o simpleprog.ll

# ${CLANG_HOME}/bin/clang++ -g -o simpleprog simpleprog.ll
# clang++ -g -o simpleprog simpleprog.ll
# clang++ -o simpleprog simpleprog.ll

clang++ -o simpleprog simpleprog.o

runlldb ./simpleprog
