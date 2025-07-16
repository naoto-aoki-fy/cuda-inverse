/* ----  STL-free core / STL in main only  ---- */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <omp.h>
#include <stdexcept>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <atlc/check_cuda.hpp>
#include <atlc/cuda.hpp>
#define DEVICE_HOST __device__ __host__
#else
#define DEVICE_HOST
#endif

/* 手書き swap（テンプレートを避ける） */
DEVICE_HOST inline void swap_int (int* a, int* b) { int t = *a; *a = *b; *b = t; }
DEVICE_HOST inline void swap_dbl (double* a, double* b) { double t = *a; *a = *b; *b = t; }

DEVICE_HOST void sync_cuda_omp() {
    #ifdef __CUDA_ARCH__
        __syncthreads();
    #else
        #pragma omp barrier
    #endif
}

/* ---------- LU 分解（部分ピボット・順次） ---------- */
DEVICE_HOST void lu_decompose_seq(uint64_t thread_num, uint64_t num_threads, double* A, int* piv, double* pAkk, bool* is_singular, int n)
{
    // __shared__ double Akk;
    if (thread_num == 0) {
        for (int i = 0; i < n; ++i) piv[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        if (thread_num == 0) {
            /* ピボット選択 */
            int    pivot = k;
            double amax  = fabs(A[k * n + k]);
            for (int i = k + 1; i < n; ++i) {
                double val = fabs(A[i * n + k]);
                if (val > amax) { amax = val; pivot = i; }
            }
            if (amax == 0.0) {
                *is_singular = true;
                // return;
                // std::fprintf(stderr, "Singular\n"); std::exit(EXIT_FAILURE);
            }

            /* 行入れ替え */
            if (pivot != k) {
                swap_int(&piv[k], &piv[pivot]);
                for (int j = 0; j < n; ++j)
                    swap_dbl(&A[k * n + j], &A[pivot * n + j]);
            }

            /* 前進消去 */
            *pAkk = A[k * n + k];
        }
        sync_cuda_omp();
        if (*is_singular) { return; }
        // #pragma omp parallel for schedule(static)
        // for (int i = k + 1; i < n; ++i) {
        for (int i = k + 1 + thread_num; i < n; i += num_threads) {
            double const Lik = (A[i * n + k] /= *pAkk);
            for (int j = k + 1; j < n; ++j)
                A[i * n + j] -= Lik * A[k * n + j];
        // }
        }
        sync_cuda_omp();
    }
}

/* ---------- 前進・後退代入 ---------- */
DEVICE_HOST void lu_solve
(
    const double* LU,
    const int* piv,
    const double* b,
    double* y,
    double* x,
    int n
)
{
    /* Pb → y */
    for (int i = 0; i < n; ++i) y[i] = b[piv[i]];

    /* Ly = Pb */
    for (int i = 0; i < n; ++i) {
        double sum = y[i];
        for (int j = 0; j <  i; ++j) sum -= LU[i * n + j] * y[j];
        y[i] = sum;
    }

    /* Ux = y */
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) sum -= LU[i * n + j] * x[j];
        x[i] = sum / LU[i * n + i];
    }
}

/* ---------- 逆行列 ---------- */
DEVICE_HOST void invert
(
    uint64_t thread_num,
    uint64_t num_threads,
    const double* A,
    double* Ainv,
    double* LU,
    int* piv,
    double* b,
    double* y,
    double* x,
    double* pAkk,
    bool* is_singular,
    int n
)
{
    /* 1. LU 分解 */
    // for (int i = 0; i < n * n; ++i) LU[i] = A[i];
    lu_decompose_seq(thread_num, num_threads, LU, piv, pAkk, is_singular, n);
    if (*is_singular) { return; }

    /* 2. n 本の方程式を解く（列ごとに単位ベクトル） */
    // #pragma omp parallel
    {
        // int tid = omp_get_thread_num();
        // int const tid = threadIdx.x;
        double* b_loc = b + thread_num * n;
        double* y_loc = y + thread_num * n;
        double* x_loc = x + thread_num * n;

        // #pragma omp for schedule(static)
        // for (int col = 0; col < n; ++col) {
        for (int col = thread_num; col < n; col += num_threads) {
            for (int i = 0; i < n; ++i) b_loc[i] = 0.0;
            b_loc[col] = 1.0;

            lu_solve(LU, piv, b_loc, y_loc, x_loc, n);

            for (int row = 0; row < n; ++row)
                Ainv[row * n + col] = x_loc[row];
        }
        // }
    }
}

#ifdef __CUDACC__
__global__
void invert_cuda
(
    const double* A,
    double* Ainv,
    double* LU,
    int* piv,
    double* b,
    double* y,
    double* x,
    bool* is_singular_device,
    int n
)
{
    __shared__ double Akk;
    __shared__ bool is_singular[1];
    if (threadIdx.x == 0) {
        *is_singular = false;
    }
    invert(threadIdx.x, blockDim.x, A, Ainv, LU, piv, b, y, x, &Akk, is_singular, n);
    if (threadIdx.x == 0 && *is_singular) {
        *is_singular_device = true;
    }
}
#endif

void invert_omp
(
    uint64_t thread_num,
    uint64_t num_threads,
    const double* A,
    double* Ainv,
    double* LU,
    int* piv,
    double* b,
    double* y,
    double* x,
    double* pAkk,
    bool* is_singular,
    int n
)
{
    invert(thread_num, num_threads, A, Ainv, LU, piv, b, y, x, pAkk, is_singular, n);
}

#ifdef __CUDACC__
int main()
{
    const int n = 60;
    int block_size = 32;
    // const int threads = omp_get_max_threads();

    /* main だけで STL を使用して領域確保 */
    std::vector<double> A(n * n);
    std::vector<double> LU(n * n);
    std::vector<double> Ainv(n * n);

    /* テスト行列生成（乱数） */
    std::srand(1234);
    for (double& v : A) {
        v = ( -1.0 + 2.0 * (std::rand() / static_cast<double>(RAND_MAX)) );
    }

    double* A_device;
    ATLC_CHECK_CUDA(cudaMalloc, &A_device, n * n * sizeof(double));
    ATLC_CHECK_CUDA(cudaMemcpyAsync, A_device, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, 0);


    double* LU_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &LU_device, n * n * sizeof(double), 0);
    double* Ainv_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &Ainv_device, n * n * sizeof(double), 0);
    int* piv_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &piv_device, n * sizeof(int), 0);
    double* b_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &b_device, block_size * n * sizeof(double), 0);
    double* y_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &y_device, block_size * n * sizeof(double), 0);
    double* x_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &x_device, block_size * n * sizeof(double), 0);

    bool* is_singular_device;
    ATLC_CHECK_CUDA(cudaMallocAsync, &is_singular_device, sizeof(bool), 0);
    bool is_singular_host[1];

    /* 逆行列計算 */

    ATLC_CHECK_CUDA(cudaMemcpyAsync, LU_device, A_device, n * n * sizeof(double), cudaMemcpyDeviceToDevice);

    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, invert_cuda, 1, block_size, 0, 0, A_device, Ainv_device, LU_device, piv_device, b_device, y_device, x_device, is_singular_device, n);

    ATLC_CHECK_CUDA(cudaMemcpyAsync, is_singular_host, is_singular_device, sizeof(bool), cudaMemcpyDeviceToHost, 0);

    ATLC_CHECK_CUDA(cudaMemcpyAsync, Ainv.data(), Ainv_device, n * n * sizeof(double), cudaMemcpyDeviceToHost, 0);

    ATLC_CHECK_CUDA(cudaStreamSynchronize, 0);

    if (*is_singular_host) {
        fprintf(stderr, "Singular\n");
        std::exit(EXIT_FAILURE);
    }

    /* Frobenius 誤差 ||I - A·A⁻¹||_F */
    double err2 = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += A[i * n + k] * Ainv[k * n + j];
            double diff = s - (i == j ? 1.0 : 0.0);
            err2 += diff * diff;
        }
    }
    std::printf("Frobenius error = %.3e\n", std::sqrt(err2));
    return 0;
}
#else
int main()
{
    const int n = 60;
    int const num_threads = omp_get_max_threads();

    /* main だけで STL を使用して領域確保 */
    std::vector<double> A(n * n);
    std::vector<double> LU;
    std::vector<double> Ainv(n * n);
    std::vector<int> piv(n);
    std::vector<double> b(num_threads * n);
    std::vector<double> y(num_threads * n);
    std::vector<double> x(num_threads * n);

    /* テスト行列生成（乱数） */
    std::srand(1234);
    for (double& v : A) {
        v = ( -1.0 + 2.0 * (std::rand() / static_cast<double>(RAND_MAX)) );
    }
    LU = A;

    double Akk;
    bool is_singular[1];
    *is_singular = false;
    #pragma omp parallel
    {
        if (num_threads != omp_get_num_threads()) {
            fprintf(stderr, "omp_get_num_threads != omp_get_max_threads\n");
            throw std::runtime_error("omp_get_num_threads != omp_get_max_threads");
        }
        uint64_t thread_num = omp_get_thread_num();
        invert_omp(thread_num, num_threads, A.data(), Ainv.data(), LU.data(), piv.data(), b.data(), y.data(), x.data(), &Akk, is_singular, n);
    }
    if (*is_singular) {
        fprintf(stderr, "Singular\n");
        std::exit(EXIT_FAILURE);
    }

    /* Frobenius 誤差 ||I - A·A⁻¹||_F */
    double err2 = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += A[i * n + k] * Ainv[k * n + j];
            double diff = s - (i == j ? 1.0 : 0.0);
            err2 += diff * diff;
        }
    }
    std::printf("Frobenius error = %.3e\n", std::sqrt(err2));
    return 0;
}
#endif
