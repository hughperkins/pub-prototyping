#!/bin/bash

LDFLAGS="-Lbuild -lstdc++" python setup.py build_ext -i

