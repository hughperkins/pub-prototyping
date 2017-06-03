#!/usr/bin/bash


TARGET=foo/bar/somefile.cu.cc

# gets a file basename and postfix, for unknown postfix
# output: BASEPATH POSTFIX
# eg input: foo/bar/somefile.cu.cc
# outputs will be:
# BASEPATH=foo/bar/somefile
# POSTFIX=.cu.cc
function split_path {
    DIRNAME=$(dirname $1)

    BASENAME=$(basename $1)

    BASEARR=(${BASENAME//./ })
    BASEPATH=${DIRNAME}/${BASEARR[0]}

    LENARR=$((${#BASEARR[@]} - 1))
    ARGREST=
    for j in $(seq 1 ${LENARR}); do {
        ARGREST="$ARGREST.${BASEARR[$j]}";
    } done
    POSTFIX=$ARGREST
}

split_path $TARGET
echo BASEPATH $BASEPATH
echo POSTFIX ${POSTFIX}
