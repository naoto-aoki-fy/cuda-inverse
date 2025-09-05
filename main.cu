// invert_complex_check_fixed.cu
// nvcc -gencode=arch=compute_61,code=sm_61 main_cusolver.cu -lcusolver -lcublas
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define CHECK_CUDA(call) { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
}
#define CHECK_CUSOLVER(call) { \
  cusolverStatus_t s = (call); \
  if (s != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSolver Error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    exit(EXIT_FAILURE); \
  } \
}
#define CHECK_CUBLAS(call) { \
  cublasStatus_t s = (call); \
  if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS Error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    exit(EXIT_FAILURE); \
  } \
}

int main() {
    const int n = 3;
    const int lda = n;

    // column-major: columns concatenated
    cuDoubleComplex h_A[n*n] = {
        make_cuDoubleComplex(1.0,  2.0), // (0,0)
        make_cuDoubleComplex(0.0,  0.0), // (1,0)
        make_cuDoubleComplex(5.0, -3.0), // (2,0)

        make_cuDoubleComplex(2.0,  0.0), // (0,1)
        make_cuDoubleComplex(1.0,  0.0), // (1,1)
        make_cuDoubleComplex(6.0,  0.0), // (2,1)

        make_cuDoubleComplex(3.0, -1.0), // (0,2)
        make_cuDoubleComplex(4.0,  2.0), // (1,2)
        make_cuDoubleComplex(0.0,  1.0)  // (2,2)
    };

    std::vector<cuDoubleComplex> h_Ainv(n*n);
    std::vector<cuDoubleComplex> h_prod(n*n);

    // handles
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // device buffers (initialize to nullptr for safe cleanup)
    cuDoubleComplex *d_A = nullptr, *d_Aorig = nullptr, *d_B = nullptr, *d_work = nullptr;
    cuDoubleComplex *d_prod = nullptr;
    int *d_ipiv = nullptr;
    int *d_info = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex)*n*n));
    CHECK_CUDA(cudaMalloc((void**)&d_Aorig, sizeof(cuDoubleComplex)*n*n)); // keep original A
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(cuDoubleComplex)*n*n)); // identity -> X (inverse)
    CHECK_CUDA(cudaMalloc((void**)&d_ipiv, sizeof(int)*n));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_prod, sizeof(cuDoubleComplex)*n*n));

    // copy A to device (both to d_A and d_Aorig)
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(cuDoubleComplex)*n*n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Aorig, h_A, sizeof(cuDoubleComplex)*n*n, cudaMemcpyHostToDevice));

    // prepare identity on device (right-hand side)
    std::vector<cuDoubleComplex> h_I(n*n);
    for (int i=0;i<n*n;++i) h_I[i] = make_cuDoubleComplex(0.0, 0.0);
    for (int i=0;i<n;++i) h_I[i + i*n] = make_cuDoubleComplex(1.0, 0.0); // column-major: i + i*n
    CHECK_CUDA(cudaMemcpy(d_B, h_I.data(), sizeof(cuDoubleComplex)*n*n, cudaMemcpyHostToDevice));

    // getrf buffer size (Z = double complex)
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(cusolverH, n, n, d_A, lda, &lwork));
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork));

    // --- Important: declare variables that might be referenced after a goto BEFORE any goto ---
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    double max_err = 0.0;

    // LU (A -> LU)
    CHECK_CUSOLVER(cusolverDnZgetrf(cusolverH, n, n, d_A, lda, d_work, d_ipiv, d_info));
    int info_h = 0;
    CHECK_CUDA(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        fprintf(stderr, "getrf failed, info=%d (matrix may be singular)\n", info_h);
        goto CLEANUP;
    }

    // solve A * X = I -> X (inverse) stored in d_B
    CHECK_CUSOLVER(cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, n, n, d_A, lda, d_ipiv, d_B, n, d_info));
    CHECK_CUDA(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        fprintf(stderr, "getrs failed, info=%d\n", info_h);
        goto CLEANUP;
    }

    // d_Aorig contains original A, d_B contains A^{-1}
    // Compute prod = A * Ainv using cuBLAS:
    // C = alpha * A * B + beta * C
    CHECK_CUBLAS(cublasZgemm(cublasH,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             (const cuDoubleComplex*)d_Aorig, lda,   // original A
                             (const cuDoubleComplex*)d_B, n,         // B = Ainv
                             &beta,
                             d_prod, n));                             // C (product)

    // Copy product back to host and compute max abs error vs identity
    CHECK_CUDA(cudaMemcpy(h_prod.data(), d_prod, sizeof(cuDoubleComplex)*n*n, cudaMemcpyDeviceToHost));

    max_err = 0.0;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            cuDoubleComplex z = h_prod[col*n + row]; // column-major
            double re = cuCreal(z);
            double im = cuCimag(z);
            double expect_re = (row == col) ? 1.0 : 0.0;
            double expect_im = 0.0;
            double dre = re - expect_re;
            double dim = im - expect_im;
            double abs_err = sqrt(dre*dre + dim*dim);
            if (abs_err > max_err) max_err = abs_err;
        }
    }

    // print product matrix
    printf("A * Ainv (row major display):\n");
    for (int row=0; row<n; ++row) {
        for (int col=0; col<n; ++col) {
            cuDoubleComplex z = h_prod[col*n + row];
            printf("% .6f%+.6fi  ", cuCreal(z), cuCimag(z));
        }
        printf("\n");
    }
    printf("max absolute error vs identity: %.6e\n", max_err);
    if (max_err < 1e-12) {
        printf("OK: result is numerically identity (tol 1e-12)\n");
    } else {
        printf("WARNING: not exact identity (max_err=%.6e)\n", max_err);
    }

CLEANUP:
    if (d_work) cudaFree(d_work);
    if (d_prod) cudaFree(d_prod);
    if (d_B) cudaFree(d_B);
    if (d_Aorig) cudaFree(d_Aorig);
    if (d_A) cudaFree(d_A);
    if (d_ipiv) cudaFree(d_ipiv);
    if (d_info) cudaFree(d_info);
    if (cublasH) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    return 0;
}
