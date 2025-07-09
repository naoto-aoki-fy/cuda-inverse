/* ----  STL-free core / STL in main only  ---- */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>   // ← main 用
#include <omp.h>
#include <cuda_runtime.h>

/* 手書き swap（テンプレートを避ける） */
__device__
inline void swap_int (int&    a, int&    b) { int    t = a; a = b; b = t; }
__device__
inline void swap_dbl (double& a, double& b) { double t = a; a = b; b = t; }

/* ---------- LU 分解（部分ピボット・順次） ---------- */
__device__
void lu_decompose_seq(double* A, int* piv, int n)
{
    __shared__ double Akk;
    if (threadIdx.x==0) {
        for (int i = 0; i < n; ++i) piv[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        if (threadIdx.x==0) {
            /* ピボット選択 */
            int    pivot = k;
            double amax  = fabs(A[k * n + k]);
            for (int i = k + 1; i < n; ++i) {
                double val = fabs(A[i * n + k]);
                if (val > amax) { amax = val; pivot = i; }
            }
            if (amax == 0.0) {
                return;
                // std::fprintf(stderr, "Singular\n"); std::exit(EXIT_FAILURE);
            }

            /* 行入れ替え */
            if (pivot != k) {
                swap_int(piv[k], piv[pivot]);
                for (int j = 0; j < n; ++j)
                    swap_dbl(A[k * n + j], A[pivot * n + j]);
            }

            /* 前進消去 */
            Akk = A[k * n + k];
        }
        __syncthreads();
        // #pragma omp parallel for schedule(static)
        // for (int i = k + 1; i < n; ++i) {
        for (int i = k + 1 + threadIdx.x; i < n; i += blockDim.x) {
            double const Lik = (A[i * n + k] /= Akk);
            for (int j = k + 1; j < n; ++j)
                A[i * n + j] -= Lik * A[k * n + j];
        // }
        }
    }
}

/* ---------- 前進・後退代入 ---------- */
__device__
void lu_solve(const double* LU,
              const int*    piv,
              const double* b,
              double*       y,
              double*       x,
              int           n)
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
__global__
void invert(const double* A,
            double*       Ainv,
            double*       LU,
            int*          piv,
            double*       b,
            double*       y,
            double*       x,
            int           n)
{
    /* 1. LU 分解 */
    // for (int i = 0; i < n * n; ++i) LU[i] = A[i];
    lu_decompose_seq(LU, piv, n);

    /* 2. n 本の方程式を解く（列ごとに単位ベクトル） */
    // #pragma omp parallel
    {
        // int tid = omp_get_thread_num();
        int const tid = threadIdx.x;
        double* b_loc = b + tid * n;
        double* y_loc = y + tid * n;
        double* x_loc = x + tid * n;

        // #pragma omp for schedule(static)
        // for (int col = 0; col < n; ++col) {
        for (int col = threadIdx.x; col < n; col += blockDim.x) {
            for (int i = 0; i < n; ++i) b_loc[i] = 0.0;
            b_loc[col] = 1.0;

            lu_solve(LU, piv, b_loc, y_loc, x_loc, n);

            for (int row = 0; row < n; ++row)
                Ainv[row * n + col] = x_loc[row];
        }
        // }
    }
}

/* ---------- テスト ---------- */
int main()
{
    const int n = 60;
    int block_size = 32;
    // const int threads = omp_get_max_threads();

    /* main だけで STL を使用して領域確保 */
    std::vector<double> A   (n * n);
    std::vector<double> LU  (n * n);
    std::vector<double> Ainv(n * n);

    /* テスト行列生成（乱数） */
    std::srand(1234);
    for (double& v : A) {
        v = ( -1.0 + 2.0 * (std::rand() / static_cast<double>(RAND_MAX)) );
    }

    

    double* A_device;
    cudaMalloc(&A_device, n * n * sizeof(double));
    cudaMemcpyAsync(A_device, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, 0);


    double* LU_device;
    cudaMallocAsync(&LU_device, n * n * sizeof(double), 0);
    double* Ainv_device;
    cudaMallocAsync(&Ainv_device, n * n * sizeof(double), 0);
    int* piv_device;
    cudaMallocAsync(&piv_device, n * sizeof(int), 0);
    double* b_device;
    cudaMallocAsync(&b_device, block_size * n * sizeof(double), 0);
    double* y_device;
    cudaMallocAsync(&y_device, block_size * n * sizeof(double), 0);
    double* x_device;
    cudaMallocAsync(&x_device, block_size * n * sizeof(double), 0);

    /* 逆行列計算 */

    cudaMemcpyAsync(LU_device, A_device, n * n * sizeof(double), cudaMemcpyDeviceToDevice);

    invert<<<1, block_size>>>(A_device, Ainv_device,
           LU_device, piv_device,
           b_device, y_device, x_device, n);

    cudaMemcpyAsync(Ainv.data(), Ainv_device, n * n * sizeof(double), cudaMemcpyDeviceToHost, 0);

    cudaStreamSynchronize(0);

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
