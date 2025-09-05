// inv_cublas_z.cu
// nvcc -gencode=arch=compute_61,code=sm_61 main_cublas.cu -lcublas
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define CHECK_CUDA(call) \
  do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while(0)

#define CHECK_CUBLAS(call) \
  do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(EXIT_FAILURE); } } while(0)

// print column-major cuDoubleComplex matrix (n x n)
void printMat(const cuDoubleComplex* A, int n){
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++){
        cuDoubleComplex v = A[i + j*n]; // column-major
        printf("(%9.5f,%9.5f) ", v.x, v.y);
      }
      printf("\n");
    }
    printf("\n");
}

int main(){
  const int n = 3;                 // 行列サイズ n x n
  const int lda = n;
  const int batchSize = 1;         // 単一行列を batched で処理

  // --- ホスト側：サンプル行列（列優先 column-major） ---
  cuDoubleComplex h_A[n*n];
  // 例: A = [[1+i, 2, 3], [4, 5+i, 6], [7, 8, 9+i]] を列優先で格納
  h_A[0 + 0*n] = make_cuDoubleComplex(1.0, 1.0); // A(0,0)
  h_A[1 + 0*n] = make_cuDoubleComplex(4.0, 0.0); // A(1,0)
  h_A[2 + 0*n] = make_cuDoubleComplex(7.0, 0.0); // A(2,0)

  h_A[0 + 1*n] = make_cuDoubleComplex(2.0, 0.0); // A(0,1)
  h_A[1 + 1*n] = make_cuDoubleComplex(5.0, 1.0); // A(1,1)
  h_A[2 + 1*n] = make_cuDoubleComplex(8.0, 0.0); // A(2,1)

  h_A[0 + 2*n] = make_cuDoubleComplex(3.0, 0.0); // A(0,2)
  h_A[1 + 2*n] = make_cuDoubleComplex(6.0, 0.0); // A(1,2)
  h_A[2 + 2*n] = make_cuDoubleComplex(9.0, 1.0); // A(2,2)

  printf("Host A (column-major):\n");
  printMat(h_A, n);

  // --- デバイスメモリ確保 ---
  cuDoubleComplex* d_A = nullptr;        // 作業用（LU→inverse）
  cuDoubleComplex* d_Aorig = nullptr;    // 元の行列を保持（積計算に使う）
  cuDoubleComplex* d_C = nullptr;        // 積の結果 (Aorig * Ainv)
  CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex)*n*n));
  CHECK_CUDA(cudaMalloc((void**)&d_Aorig, sizeof(cuDoubleComplex)*n*n));
  CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(cuDoubleComplex)*n*n));

  // コピー: d_A にコピーしてその場で LU/逆行列計算をする（in-place）
  CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(cuDoubleComplex)*n*n, cudaMemcpyHostToDevice));
  // d_Aorig にもコピーして、inverse と掛ける時に使う
  CHECK_CUDA(cudaMemcpy(d_Aorig, h_A, sizeof(cuDoubleComplex)*n*n, cudaMemcpyHostToDevice));

  // デバイス上の行列ポインタ配列（batched API用）
  cuDoubleComplex* h_Aptr[batchSize];
  h_Aptr[0] = d_A;
  cuDoubleComplex** d_Aptr = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&d_Aptr, sizeof(cuDoubleComplex*) * batchSize));
  CHECK_CUDA(cudaMemcpy(d_Aptr, h_Aptr, sizeof(cuDoubleComplex*) * batchSize, cudaMemcpyHostToDevice));

  // pivots と info（デバイス上）
  int* d_pivots = nullptr;   // pivot array: length = n * batchSize
  int* d_info   = nullptr;   // info array: length = batchSize
  CHECK_CUDA(cudaMalloc((void**)&d_pivots, sizeof(int) * n * batchSize));
  CHECK_CUDA(cudaMalloc((void**)&d_info,   sizeof(int) * batchSize));

  // --- cuBLAS ハンドル ---
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // LU 分解（in-place）
  CHECK_CUBLAS(cublasZgetrfBatched(handle, n, (cuDoubleComplex**)d_Aptr, lda, d_pivots, d_info, batchSize));

  // info をホストに戻してエラー確認
  int h_info[batchSize];
  CHECK_CUDA(cudaMemcpy(h_info, d_info, sizeof(int)*batchSize, cudaMemcpyDeviceToHost));
  for(int i=0;i<batchSize;i++){
    if(h_info[i] != 0){
      fprintf(stderr, "getrfBatched failed for batch %d: info=%d\n", i, h_info[i]);
      // cleanup before exit
      cublasDestroy(handle);
      cudaFree(d_Aptr);
      cudaFree(d_A);
      cudaFree(d_Aorig);
      cudaFree(d_pivots);
      cudaFree(d_info);
      cudaFree(d_C);
      exit(EXIT_FAILURE);
    }
  }

  // 逆行列計算（in-place: d_A が逆行列に置き換わる）
  // 注意: getriBatched のシグネチャは CUDA バージョンによって差があるので
  // コンパイル時にヘッダを確認してください。ここでは一般的な形を使います。
  CHECK_CUBLAS(cublasZgetriBatched(handle, n, (cuDoubleComplex**)d_Aptr, lda, d_pivots, (cuDoubleComplex**)d_Aptr, lda, d_info, batchSize));

  // info 再チェック
  CHECK_CUDA(cudaMemcpy(h_info, d_info, sizeof(int)*batchSize, cudaMemcpyDeviceToHost));
  for(int i=0;i<batchSize;i++){
    if(h_info[i] != 0){
      fprintf(stderr, "getriBatched failed for batch %d: info=%d\n", i, h_info[i]);
      // cleanup before exit
      cublasDestroy(handle);
      cudaFree(d_Aptr);
      cudaFree(d_A);
      cudaFree(d_Aorig);
      cudaFree(d_pivots);
      cudaFree(d_info);
      cudaFree(d_C);
      exit(EXIT_FAILURE);
    }
  }

  // d_A は逆行列になっている。これを d_Ainv と見なす。
  cuDoubleComplex* d_Ainv = d_A;

  // --- 元行列 * 逆行列 を計算 (C = Aorig * Ainv) ---
  cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

  // cublasZgemm(handle, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)
  CHECK_CUBLAS(cublasZgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           n, n, n,
                           &alpha,
                           d_Aorig, lda,
                           d_Ainv,  lda,
                           &beta,
                           d_C,     lda));

  // 結果をホストにコピーして表示
  cuDoubleComplex h_inv[n*n];
  cuDoubleComplex h_prod[n*n];
  CHECK_CUDA(cudaMemcpy(h_inv, d_Ainv, sizeof(cuDoubleComplex)*n*n, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_prod, d_C, sizeof(cuDoubleComplex)*n*n, cudaMemcpyDeviceToHost));

  printf("Inverse (from device):\n");
  printMat(h_inv, n);

  printf("Product A * Ainv (should be identity):\n");
  printMat(h_prod, n);

  // 誤差評価（最大ノルム）
  double max_err = 0.0;
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      double expect_re = (i==j) ? 1.0 : 0.0;
      double expect_im = 0.0;
      cuDoubleComplex v = h_prod[i + j*n];
      double dr = v.x - expect_re;
      double di = v.y - expect_im;
      double abs_err = sqrt(dr*dr + di*di);
      if(abs_err > max_err) max_err = abs_err;
    }
  }
  printf("max abs error vs I: %.6e\n", max_err);

  // cleanup
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_Aptr));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_Aorig));
  CHECK_CUDA(cudaFree(d_pivots));
  CHECK_CUDA(cudaFree(d_info));
  CHECK_CUDA(cudaFree(d_C));

  return 0;
}
