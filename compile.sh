#!/bin/bash

SM_VER=61 # GTX1060
# SM_VER=80 # A100
# SM_VER=90 # H100

OPTARG=(-O3 -std=c++17 --cudart=shared -lcurand)

if [[ inverse.cu -nt main.exe ]]; then

    nvcc \
        -gencode=arch=compute_${SM_VER},code=sm_${SM_VER} \
        "${OPTARG[@]}" \
        inverse.cu -o main.exe
    
    nvcc_status_code=$?

    if [[ $nvcc_status_code != 0 ]]; then
        exit
    fi

fi

./main.exe