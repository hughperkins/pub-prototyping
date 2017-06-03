#!/bin/bash

# eg test with:
# bash ~/prototyping/bash/testgetopt.sh -M -iquote foobar -iquote 'another iquote' -isystem foobar -o blah.o foobar.cu -gencode=arch=compute_30,\"code=sm30,compute_30\"

# LENSUB1=$((${#arr1[@]} - 1))
# for j in $(seq 1 ${LENSUB1}); do {
#     if [[ $j != 1 ]]; then {
#         ARGREST="$ARGREST="
#     } fi
#     ARGREST="$ARGREST${arr1[$j]}";
# } done
# echo ARGREST $ARGREST

PASSTHRU=
while [[ x$1 != x ]]; do {
    # echo 1 $1
    arr1=(${1//=/ })
    THISARG=${arr1[0]}
    # echo THISARG ${THISARG}
    case $THISARG in
        -M)
            PASSTHRU="$PASSTHRU -M"
            ;;
        -o)
            OUTPATH=$2
            shift
            ;;
        -iquote)
            PASSTHRU="$PASSTHRU $1 $2"
            shift
            ;;
        -isystem)
            PASSTHRU="$PASSTHRU $1 $2"
            shift
            ;;
        -gencode)
            PASSTHRU="$PASSTHRU $1"
            ;;
        *)
            if [[ x$INFILES != x ]]; then {
                INFILES="$INFILES $1"
            } else {
                INFILES="$1"
            } fi
            ;;
    esac
    # esac
    shift
} done

echo OUTPATH [$OUTPATH]
echo INFILES ["$INFILES"]
echo PASSTHRU $PASSTHRU

# while getopts MNgco:I:D:i: opt; do
#     # echo args "$@"
#     # echo OPTIND $OPTIND
#     # echo ${$OPTIND}
#     eval "THIS=\${$OPTIND}"
#     # echo THIS $THIS
#     # echo loop 1 $1 2 $2 3 $3 4 $4
#     # set | grep OPT
#     case $opt in
#     I)
#         INCLUDES="$INCLUDES -I$OPTARG"
#         ;;
#     o)
#         OUTPATH=$OPTARG
#         echo OUTPATH $OUTPATH
#         ;;
#     c)
#         COMPILE=1
#         ;;
#     g)
#         OPT_G=-g
#         ;;
#     M)
#         echo warning: ignoring -M arg
#         ;;
#     N)
#         echo warning: ignoring -N arg
#         ;;
#     D)
#         echo warning: ignoring -D arg: $OPTARG
#         ;;
#     i)
#         IARG="-i$OPTARG $THIS"
#         echo warning: ignoring -i arg: $IARG;
#         shift;
#         # OPTIND=$(($OPTIND + 1)) 
#         ;;
#     U)
#         echo warning: ignoring -U arg: $OPTARG
#         ;;
#     esac
# done
# shift $((OPTIND - 1))

# echo args "$@"
