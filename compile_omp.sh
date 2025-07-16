#!/bin/bash

SOURCE=inverse_002.cpp
EXEFN=main_omp.exe

OPTARG=(-O3 -fopenmp -std=c++17 )

if [[ "${SOURCE}" -nt "${EXEFN}" ]]; then

    g++ "${OPTARG[@]}" "${SOURCE}" -o "${EXEFN}"
    
    cc_status_code=$?

    if [[ $cc_status_code != 0 ]]; then
        exit
    fi

fi

./"${EXEFN}"