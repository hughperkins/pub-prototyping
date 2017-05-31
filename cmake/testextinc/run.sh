#!/bin/bash

rm -Rf install
rm -Rf local_install
rm -Rf build
mkdir build || exit 1
cd build || exit 1
echo     
echo     
echo     
echo     
echo     
echo     
set -e
set -x
cmake .. -DCMAKE_INSTALL_PREFIX=${PWD}/../install
echo "DONE CMAKE =================="
#make -j 4
echo "DONE MAKE =================="
set +x
echo
echo
echo
set -x
make -j 4 install

