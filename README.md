# CUDA Inverse Example

This repository provides a small example that computes the inverse of a 25×25 matrix from a single source file. When compiled with CUDA the program launches a single block of 25 threads to perform the LU decomposition and inversion on the GPU. The same code can also be compiled for CPU execution with OpenMP.

A lightweight set of headers in the `atlc` directory supplies simple CUDA error-checking helpers.

## Building

Use the provided Makefile to build and run the samples:

```bash
make cuda        # build and run the CUDA version
make omp         # build and run the OpenMP version
# override the GPU architecture if needed
make cuda SM_VER=80
make clean       # remove generated executables
```

The CUDA build produces `main_cuda.exe` and the OpenMP build produces `main_omp.exe`. Each target runs the corresponding executable after a successful compilation.
