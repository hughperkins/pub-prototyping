#!/bin/bash

set -e
set -x

clang++-3.8 -o build/testbitcast testbitcast.cpp
build/testbitcast
