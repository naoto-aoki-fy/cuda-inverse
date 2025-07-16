#!/bin/bash

SM_VER=61 # GTX1060
# SM_VER=80 # A100
# SM_VER=90 # H100

SOURCE=inverse_002.cpp
EXEFN=main_cuda.exe

OPTARG=(-O3 -Xcompiler=-fopenmp -std=c++17 --cudart=shared -x cu)

if [[ "${SOURCE}" -nt "${EXEFN}" ]]; then

    nvcc \
        -gencode=arch=compute_${SM_VER},code=sm_${SM_VER} \
        "${OPTARG[@]}" \
        "${SOURCE}" -o "${EXEFN}"
    
    nvcc_status_code=$?

    if [[ $nvcc_status_code != 0 ]]; then
        exit
    fi

fi

./"${EXEFN}"