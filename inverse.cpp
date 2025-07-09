/* ----  STL-free core / STL in main only  ---- */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>   // ← main 用
#include <omp.h>

/* 手書き swap（テンプレートを避ける） */
inline void swap_int (int&    a, int&    b) { int    t = a; a = b; b = t; }
inline void swap_dbl (double& a, double& b) { double t = a; a = b; b = t; }

/* ---------- LU 分解（部分ピボット・順次） ---------- */
void lu_decompose_seq(double* A, int* piv, int n)
{
    for (int i = 0; i < n; ++i) piv[i] = i;

    for (int k = 0; k < n; ++k) {
        /* ピボット選択 */
        int    pivot = k;
        double amax  = std::fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double val = std::fabs(A[i * n + k]);
            if (val > amax) { amax = val; pivot = i; }
        }
        if (amax == 0.0) { std::fprintf(stderr, "Singular\n"); std::exit(EXIT_FAILURE); }

        /* 行入れ替え */
        if (pivot != k) {
            swap_int(piv[k], piv[pivot]);
            for (int j = 0; j < n; ++j)
                swap_dbl(A[k * n + j], A[pivot * n + j]);
        }

        /* 前進消去 */
        double Akk = A[k * n + k];
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; ++i) {
            double Lik = (A[i * n + k] /= Akk);
            for (int j = k + 1; j < n; ++j)
                A[i * n + j] -= Lik * A[k * n + j];
        }
    }
}

/* ---------- 前進・後退代入 ---------- */
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
    for (int i = 0; i < n * n; ++i) LU[i] = A[i];
    lu_decompose_seq(LU, piv, n);

    /* 2. n 本の方程式を解く（列ごとに単位ベクトル） */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* b_loc = b + tid * n;
        double* y_loc = y + tid * n;
        double* x_loc = x + tid * n;

        #pragma omp for schedule(static)
        for (int col = 0; col < n; ++col) {
            for (int i = 0; i < n; ++i) b_loc[i] = 0.0;
            b_loc[col] = 1.0;

            lu_solve(LU, piv, b_loc, y_loc, x_loc, n);

            for (int row = 0; row < n; ++row)
                Ainv[row * n + col] = x_loc[row];
        }
    }
}

/* ---------- テスト ---------- */
int main()
{
    const int n = 25;
    const int threads = omp_get_max_threads();

    /* main だけで STL を使用して領域確保 */
    std::vector<double> A   (n * n);
    std::vector<double> LU  (n * n);
    std::vector<double> Ainv(n * n);
    std::vector<int   > piv (n);
    std::vector<double> b(threads * n), y(threads * n), x(threads * n);

    /* テスト行列生成（乱数） */
    std::srand(1234567);
    for (double& v : A)
        v = -1.0 + 2.0 * (std::rand() / static_cast<double>(RAND_MAX));

    /* 逆行列計算 */
    invert(A.data(), Ainv.data(),
           LU.data(), piv.data(),
           b.data(), y.data(), x.data(), n);

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
