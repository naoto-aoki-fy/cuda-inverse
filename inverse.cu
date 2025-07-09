// inverse25.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#include "check_curand.hpp"

#define NS 25          // 行列サイズ
#define BLOCK NS       // スレッド数＝列数

// カーネル ------------------------------------------------------------------
__global__ void invert25(const double* __restrict__ A_in,
                         double*       __restrict__ A_inv)
{
    __shared__ double A[NS][NS];
    __shared__ double Inv[NS][NS];
    __shared__ int    piv[NS];

    const int col = threadIdx.x;  // 0～24

    // 1. グローバル → 共有メモリにロード
    for (int row = 0; row < NS; ++row) {
        A[row][col]   = A_in[row * NS + col];
        Inv[row][col] = (row == col) ? 1.0 : 0.0;
    }
    __syncthreads();

    // 2. LU 分解（Crout, 部分ピボット）
    for (int k = 0; k < NS; ++k) {

        // 2.1 ピボット選択（共有メモリ上で列 k の最大値行を探す）
        double maxVal = 0.0;
        int    maxRow = k;
        double val    = (col >= k) ? fabs(A[col][k]) : 0.0; // pivot search only for rows >= k
        // warp-wide reduce (BLOCK ≤ 32 のため単一 warp で完結)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_xor_sync(0xffffffff, val, offset);
            int    oRow  = __shfl_xor_sync(0xffffffff, col, offset);
            if (other > val) {
                val    = other;
                maxRow = oRow;
            }
        }
        // スレッド0が採用
        if (col == 0) {
            piv[k] = maxRow;
            if (maxRow != k) {
                for (int j = 0; j < NS; ++j) {
                    double tmp = A[k][j]; A[k][j] = A[maxRow][j]; A[maxRow][j] = tmp;
                    tmp = Inv[k][j];      Inv[k][j] = Inv[maxRow][j]; Inv[maxRow][j] = tmp;
                }
            }
        }
        __syncthreads();

        // 2.2 L 列更新（行 k+1..NS-1, 列 k）
        double pivot = A[k][k];
        double aik   = A[col][k];
        if (col > k)  A[col][k] = aik /= pivot;
        __syncthreads();

        // 2.3 U 部分更新（下三角ブロックを列方向並列で）
        if (col > k) {
            double lik = aik;   // A[col][k] after division
            for (int j = k + 1; j < NS; ++j)
                A[col][j] -= lik * A[k][j];
            for (int j = 0; j < NS; ++j)
                Inv[col][j] -= lik * Inv[k][j];
        }
        __syncthreads();
    }

    // 3. 25 列同時の前進 + 後退代入
    // 各スレッドは自身の列 (col) を Inv から取り出し、y, x を逐行更新
    // 前進 (Ly = e_col)
    for (int i = 0; i < NS; ++i) {
        double sum = Inv[i][col];
        for (int j = 0; j < i; ++j)
            sum -= A[i][j] * Inv[j][col];
        Inv[i][col] = sum;   // y
    }
    // 後退 (Ux = y)
    for (int i = NS - 1; i >= 0; --i) {
        double sum = Inv[i][col];
        for (int j = i + 1; j < NS; ++j)
            sum -= A[i][j] * Inv[j][col];
        Inv[i][col] = sum / A[i][i];
    }
    __syncthreads();

    // 4. 共有 → グローバルへ書き戻し
    for (int row = 0; row < NS; ++row)
        A_inv[row * NS + col] = Inv[row][col];
}

// ----------------------------------------------------------------------------
void checkCuda(cudaError_t e, const char* msg)
{
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

int main()
{
    constexpr int Nbytes = NS * NS * sizeof(double);
    double *hA   = (double*)malloc(Nbytes);
    double *hInv = (double*)malloc(Nbytes);

    // テスト行列（乱数）
    srand(1234567);
    for (int i = 0; i < NS * NS; ++i) hA[i] = rand() * (1.0 / RAND_MAX);

    double *dA, *dInv;
    checkCuda(cudaMalloc(&dA, Nbytes), "malloc dA");
    checkCuda(cudaMalloc(&dInv, Nbytes), "malloc dInv");
    checkCuda(cudaMemcpy(dA, hA, Nbytes, cudaMemcpyHostToDevice), "memcpy H→D");

    // カーネル呼び出し
    invert25<<<1, BLOCK>>>(dA, dInv);
    checkCuda(cudaGetLastError(), "kernel");

    checkCuda(cudaMemcpy(hInv, dInv, Nbytes, cudaMemcpyDeviceToHost), "memcpy D→H");

    // 検算：A × A_inv ≈ I
    double maxErr = 0.0;
    for (int i = 0; i < NS; ++i)
        for (int j = 0; j < NS; ++j) {
            double s = 0.0;
            for (int k = 0; k < NS; ++k) s += hA[i * NS + k] * hInv[k * NS + j];
            maxErr = fmax(maxErr, fabs(s - (i == j ? 1.0 : 0.0)));
        }
    printf("max |A·A⁻¹ - I| = %.3e\n", maxErr);

    cudaFree(dA);
    cudaFree(dInv);
    free(hA);
    free(hInv);
    return 0;
}
