#!/bin/bash

set -e
set -x

CC=gcc
if [[ $(uname) == Darwin ]]; then {
    CC=clang
} fi

basename=$(basename $1)
echo basename $basename

${CC} -o $basename $1
./$basename
