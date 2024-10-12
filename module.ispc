#define TILE_SIZE 32

// 计算矩阵乘积A B的一行结果，存入**向量**C中
// A为向量（原矩阵的一行）
export inline void matmulRow(uniform float A[], uniform float B[], uniform float C[],
                    uniform int M, uniform int N, uniform int K, uniform int m) {
    uniform float sumTile[TILE_SIZE], oneAVal;

    for (uniform int k0 = 0; k0 < K; k0 += TILE_SIZE) {
        uniform int size = min(TILE_SIZE, K - k0);
        // SPMD "horizontally" over TILE dimension:
        foreach (ki = 0 ... size) {
            // No scatter required.
            sumTile[ki] = 0.0f;
        }

        // Loop over the the matrix N dimension:
        for (uniform int n = 0; n < N; n++) {
            oneAVal = A[n];
            // SPMD iterate over the TILE dimension, but within for loop nest:
            foreach (kt = 0 ... size) {
                // Note, no gather required.
                varying float matB  = B[n * K + k0 + kt];
                // Pure SIMD FMAC:
                sumTile[kt] += oneAVal * matB;
            }
        }

        // SPMD "horizontally" again over TILE dimension:
        foreach (ki = 0 ... size) {
            // Note, no scatter required.
            C[k0 + ki] = sumTile[ki];
        }
    }
}

// 计算矩阵乘积A B，结果存入矩阵C
// from: https://github.com/ispc/ispc/blob/main/examples/cpu/sgemm/SGEMM_kernels.ispc
export void matmul(uniform float A[], uniform float B[], uniform float C[],
                    uniform int M, uniform int N, uniform int K) {
    for (uniform int m = 0; m < M; m++) {
        matmulRow(A + m * N, B, C + m * K, M, N, K, m);
    }
}

// 计算矩阵乘积A B^T的一行结果，存入**向量**C中
// A为向量（原矩阵的一行）
export inline void matmul_TRow(uniform float A[], uniform float B[], uniform float C[],
                    uniform int M, uniform int N, uniform int K, uniform int m) {
    uniform float sumTile[TILE_SIZE], oneAVal;

    for (uniform int k0 = 0; k0 < K; k0 += TILE_SIZE) {
            uniform int size = min(TILE_SIZE, K - k0);
            foreach (ki = 0 ... size) {
                sumTile[ki] = 0.0f;
            }

            for (uniform int n = 0; n < N; n++) {
                oneAVal = A[n];
                foreach (kt = 0 ... size) {
                    // gather required. A、B必然有一方需要按列读取
                    #pragma ignore warning(perf)
                    varying float matB  = B[(k0 + kt) * N + n];
                    sumTile[kt] += oneAVal * matB;
                }
            }

            foreach (ki = 0 ... size) {
                C[k0 + ki] = sumTile[ki];
            }
        }
}

// 计算矩阵乘积A B^T，结果存入矩阵C
export void matmul_T(uniform float A[], uniform float B[], uniform float C[],
                    uniform int M, uniform int N, uniform int K) {
    for (uniform int m = 0; m < M; m++) {
        matmul_TRow(A + m * N, B, C + m * K, M, N, K, m);
    }
}

export inline void safeVecSoftmax(uniform float A[], uniform int N) {
    uniform float max_val, sum;
    max_val = 0.f;
    foreach (n = 0 ... N) {
        max_val = max(max_val, reduce_max(A[n]));
    }

    sum = 0.f;
    foreach (n = 0 ... N) {
        float val = exp(A[n] - max_val);
        A[n] = val;
        sum += reduce_add(val);
    }

    foreach (n = 0 ... N) {
        A[n] /= sum;
    }
}

// 计算softmax
export void safeSoftmax(uniform float A[], uniform int M, uniform int N) {
    for (uniform int m = 0; m < M; ++m) {
        safeVecSoftmax(A + m * N, N);
    }
}

# define BLOCK_SIZE 16
// 分块计算矩阵乘积A B，假设块大小恒为16
export void blockMatmul(uniform float A[], uniform float B[], uniform float C[],
                    uniform int M, uniform int N, uniform int K,
                    uniform int m1, uniform int n1, uniform int k1,
                    uniform int m2, uniform int n2, uniform int k2) {
    uniform float sumTile[BLOCK_SIZE], oneAVal;

    for (uniform int m = m1; m < m2; m++) {
        uniform int size = k2 - k1;
        foreach (ki = 0 ... size) {
            sumTile[ki] = C[m * K + k1 + ki];
        }

        for (uniform int n = n1; n < n2; n++) {
            oneAVal = A[m * N + n];
            foreach (kt = 0 ... size) {
                varying float matB  = B[n * K + k1 + kt];
                sumTile[kt] += oneAVal * matB;
            }
        }

        foreach (ki = 0 ... size) {
            C[m * K + k1 + ki] = sumTile[ki];
        }
    }
}

// 分块计算矩阵乘积A B^T，假设块大小恒为16
export void blockMatmul_T(uniform float A[], uniform float B[], uniform float C[],
                    uniform int M, uniform int N, uniform int K,
                    uniform int m1, uniform int n1, uniform int k1,
                    uniform int m2, uniform int n2, uniform int k2) {
    uniform float sumTile[BLOCK_SIZE], oneAVal;

    for (uniform int m = m1; m < m2; m++) {
        uniform int size = k2 - k1;
        foreach (ki = 0 ... size) {
            sumTile[ki] = C[m * K + k1 + ki];
        }

        for (uniform int n = n1; n < n2; n++) {
            oneAVal = A[m * N + n];
            foreach (kt = 0 ... size) {
                #pragma ignore warning(perf)
                varying float matB  = B[(k1 + kt) * N + n];
                sumTile[kt] += oneAVal * matB;
            }
        }

        foreach (ki = 0 ... size) {
            C[m * K + k1 + ki] = sumTile[ki];
        }
    }
}

// 矩阵的exp
export void matexp(uniform float A[], uniform float B[], uniform int M, uniform int N) {
    for (uniform int m = 0; m < M; ++m)
        foreach (n = 0 ... N)
            B[m * N + n] = exp(A[m * N + n]);
}

// 计算矩阵每行的和
export void rowsum(uniform float A[], uniform float B[], uniform int M, uniform int N) {
    for (uniform int m = 0; m < M; ++m) {
        uniform float val = 0.f;
        foreach (n = 0 ... N)
            val += reduce_add(A[m * N + n]);
        B[m] = val;
    }
}

// 计算两个向量和
export void vecsum(uniform float A[], uniform float B[], uniform float C[], uniform int N) {
    foreach (n = 0 ... N)
        C[n] = A[n] + B[n];
}

// 计算C = (v1 * C + A B) / v2
export void flashAttentionHelper(uniform float A[], uniform float B[], uniform float C[],
                    uniform float v1[], uniform float v2[],
                    uniform int M, uniform int N, uniform int K) {
    uniform float sumTile[TILE_SIZE], oneAVal;

    for (uniform int m = 0; m < M; m++) {
        for (uniform int k0 = 0; k0 < K; k0 += TILE_SIZE) {
            uniform int size = min(TILE_SIZE, K - k0);
            foreach (ki = 0 ... size) {
                sumTile[ki] = 0.0f;
            }

            for (uniform int n = 0; n < N; n++) {
                oneAVal = A[m * N + n];
                foreach (kt = 0 ... size) {
                    varying float matB  = B[n * K + k0 + kt];
                    sumTile[kt] += oneAVal * matB;
                }
            }

            foreach (ki = 0 ... size) {
                C[m * K + k0 + ki] = (v1[m] * C[m * K + k0 + ki] + sumTile[ki]) / v2[m];
            }
        }
    }
}
