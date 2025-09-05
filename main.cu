// invert_block_cusolverdx.cu
#include <cuda_runtime.h>
#include <cusolverdx.hpp>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#ifndef TARGET_SM   // 対象GPUのSM。A100/RTX30系=800, Hopper=900, Ada=890 など
#define TARGET_SM 900
#endif

// 問題サイズ
constexpr int N = 16;

// cuSolverDx の関数ディスクリプタ: 一般行列 A の LU(部分ピボット) + 連立方程式解法(GESV)
// B の列数 K=N にして B=I を入れることで A^{-1} を得る
using GESV = decltype(
    cusolverdx::Size<N, N, N>() +
    cusolverdx::Precision<double>() +
    cusolverdx::Type<cusolverdx::type::real>() +
    cusolverdx::Arrangement<cusolverdx::col_major>() + // A, B とも列メモリ
    cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>() +
    cusolverdx::SM<TARGET_SM>() +
    cusolverdx::Block());

// double 用の atomicMax（非負値向け）
__device__ inline double atomicMaxDouble(double* addr, double val) {
    auto uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *uaddr, assumed;
    do {
        assumed = old;
        double old_d = __longlong_as_double(assumed);
        double new_d = fmax(old_d, val);
        old = atomicCAS(uaddr, assumed, __double_as_longlong(new_d));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// 1ブロック内だけで A^{-1} を求め、A*A^{-1} を計算して最大誤差を返す
template<class SOLVER>
__global__ void invert_and_check_kernel(const double* __restrict__ A_g, int lda_g,
                                        double* __restrict__ Ainv_g, int lda_inv_g,
                                        double* __restrict__ Prod_g, int ldc_g,
                                        typename SOLVER::status_type* info_out,
                                        double* max_err_out)
{
    using namespace cusolverdx;
    constexpr int n        = SOLVER::n_size;   // = N
    constexpr int lda_smem = SOLVER::lda;      // 共有メモリ上の leading dimension
    constexpr int ldb_smem = SOLVER::ldb;      // 共有メモリ上の leading dimension (B)

    extern __shared__ __align__(sizeof(double)) unsigned char smem_raw[];
    double* As = reinterpret_cast<double*>(smem_raw);                           // A（共有）
    double* Bs = As + size_t(lda_smem) * n;                                     // B（共有, 初期は I, 結果は A^{-1}）
    // 以降の領域は Solver のワークスペースとして cuSolverDx が使う（触らない）

    // --- A を gmem -> smem へロード（列ごとに分担）
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        for (int row = 0; row < n; ++row) {
            As[row + col * lda_smem] = A_g[row + col * lda_g];
        }
    }
    // --- B を単位行列に初期化（共有メモリ）
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        for (int row = 0; row < n; ++row) {
            Bs[row + col * ldb_smem] = (row == col) ? 1.0 : 0.0;
        }
    }
    __syncthreads();

    // 部分ピボット配列（共有メモリ）
    // __shared__ typename SOLVER::ipiv_data_type ipiv[n];
    __shared__ int ipiv[n];

    if (threadIdx.x == 0) {
        *info_out = 0;
        if (max_err_out) *max_err_out = 0.0;
    }
    __syncthreads();

    // --- 共有メモリ上で GESV 実行: A * X = I（B）を解いて X=B <- A^{-1}
    SOLVER().execute(As, lda_smem, ipiv, Bs, ldb_smem, info_out);
    __syncthreads();

    // --- A^{-1} を gmem へ書き戻し
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        for (int row = 0; row < n; ++row) {
            Ainv_g[row + col * lda_inv_g] = Bs[row + col * ldb_smem];
        }
    }
    __syncthreads();

    // --- C = A * A^{-1} を計算（A は gmem, A^{-1} は smem(Bs)のまま利用）
    double local_max = 0.0;
    for (int t = threadIdx.x; t < n * n; t += blockDim.x) {
        int row = t % n;
        int col = t / n;
        double sum = 0.0;
        #pragma unroll
        for (int k = 0; k < n; ++k) {
            sum += A_g[row + k * lda_g] * Bs[k + col * ldb_smem];
        }
        if (Prod_g) {
            Prod_g[row + col * ldc_g] = sum;  // 検証用に書き出したい場合
        }
        double target = (row == col) ? 1.0 : 0.0;
        local_max = fmax(local_max, fabs(sum - target));
    }

    // warp内縮約 → ブロックで最大値を1回だけatomic
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmax(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));
    }
    if ((threadIdx.x & 31) == 0 && max_err_out) {
        atomicMaxDouble(max_err_out, local_max);
    }
}

// 簡易チェック用
#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while(0)

int main() {
    using namespace cusolverdx;

    // 乱数で A を生成し、対角を少し大きくして可逆に
    std::vector<double> A(N * N), Ainv(N * N), Prod(N * N);
    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            A[i + j * N] = dist(mt);
        }
        A[j + j * N] += N; // 対角大にして非特異に
    }

    // デバイス確保
    double *dA = nullptr, *dInv = nullptr, *dProd = nullptr, *dMaxErr = nullptr;
    int *dInfo = nullptr;
    CHECK_CUDA(cudaMalloc(&dA,   sizeof(double) * N * N));
    CHECK_CUDA(cudaMalloc(&dInv, sizeof(double) * N * N));
    CHECK_CUDA(cudaMalloc(&dProd,sizeof(double) * N * N));
    CHECK_CUDA(cudaMalloc(&dMaxErr, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dInfo, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dInv,  0, sizeof(double) * N * N));
    CHECK_CUDA(cudaMemset(dProd, 0, sizeof(double) * N * N));
    CHECK_CUDA(cudaMemset(dMaxErr, 0, sizeof(double)));

    // 共有メモリ必要量と推奨BlockDimはディスクリプタから取得
    const size_t smem_bytes = GESV::shared_memory_size;
    const dim3   blockDim   = GESV::block_dim; // 推奨スレッド数

    // カーネル起動（1ブロック）
    invert_and_check_kernel<GESV><<<1, blockDim, smem_bytes>>>(
        dA, N, dInv, N, dProd, N, dInfo, dMaxErr);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // 結果確認
    int info;
    double max_err;
    CHECK_CUDA(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&max_err, dMaxErr, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Ainv.data(), dInv, sizeof(double) * N * N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Prod.data(), dProd, sizeof(double) * N * N, cudaMemcpyDeviceToHost));

    if (info != 0) {
        printf("cuSolverDx GESV failed: info = %d (A is singular or ill-conditioned)\n", info);
    } else {
        printf("max |A*A^{-1} - I|_∞ = %.3e\n", max_err);
    }

    cudaFree(dA); cudaFree(dInv); cudaFree(dProd); cudaFree(dInfo); cudaFree(dMaxErr);
    return 0;
}
