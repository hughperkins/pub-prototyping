#!/bin/bash

set -x
set -e
set -o pipefail

function error {
    echo error stuff here
    echo out.txt:
    cat out.txt

    echo finished handling error

    exit 1
}

if [[ -f out.txt ]]; then {
    rm out.txt
} fi

bash child.sh | tee out.txt || error

echo out.txt:
cat out.txt

echo end of script
