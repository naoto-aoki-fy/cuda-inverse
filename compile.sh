#!/bin/bash

set -x

if [[ inverse.cpp -nt inverse ]]; then

    g++ -O3 -fopenmp -std=c++17 inverse.cpp -o inverse
    
    cc_status_code=$?

    if [[ $cc_status_code != 0 ]]; then
        exit
    fi

fi

./inverse