#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cmath>

// Uncomment for ISPC
#include "module_ispc.h"
using namespace ispc;
#define ISPC

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int x, int y, const int sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int x, int y, const int sizeX, float val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int x, int y, int z, int b,
        const int sizeX, const int sizeY, const int sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int x, int y, int z, int b,
        const int sizeX, const int sizeY, const int sizeZ, float val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; ++b) { // batch
        for (int h = 0; h < H; ++h) { // head
            // 计算：Q[b][h] * K[b][h]^T
#ifndef ISPC
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float val = 0;
                    for (int k = 0; k < d; ++k)
                        val += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }
#else
            int step = b * H * N * d + h * N * d;
            matmul_T(Q.data() + step, K.data() + step, QK_t.data(), N, d, N);
#endif

            // 计算softmax
#ifndef ISPC
            for (int i = 0; i < N; ++i) {
                float max_val = 0;
                for (int j = 0; j < N; ++j)
                    max_val = std::max(max_val, twoDimRead(QK_t, i, j, N));

                float l = 0;
                for (int j = 0; j < N; ++j) {
                    float val = exp(twoDimRead(QK_t, i, j, N) - max_val);
                    twoDimWrite(QK_t, i, j, N, val);
                    l += val;
                }

                for (int j = 0; j < N; ++j) {
                    float val = twoDimRead(QK_t, i, j, N) / l;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }
#else
            safeSoftmax(QK_t.data(), N, N);
#endif

            // 计算QK^t * V[b][h]
#ifndef ISPC
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < d; ++j) {
                    float val = 0;
                    for (int k = 0; k < N; ++k)
                        val += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }
#else
            matmul(QK_t.data(), V.data() + step, O.data() + step, N, N, d);
#endif
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    static const int L = 16;  // 64B / 4B
    for (int b = 0; b < B; ++b) { // batch
        for (int h = 0; h < H; ++h) { // head
            std::fill(QK_t.begin(), QK_t.end(), 0);
            int step = b * H * N * d + h * N * d;
            // 分块计算：Q[b][h] * K[b][h]^T
            for (int i = 0; i < N; i += L) {
                for (int j = 0; j < N; j += L) {
                    for (int k = 0; k < d; k += L) {
                        int max_sub_i = std::min(i + L, N);
                        int max_sub_j = std::min(j + L, N);
                        int max_sub_k = std::min(k + L, d);
#ifndef ISPC
                        for (int sub_i = i; sub_i < max_sub_i; ++sub_i) {
                            for (int sub_j = j; sub_j < max_sub_j; ++sub_j) {
                                float val = twoDimRead(QK_t, sub_i, sub_j, N);
                                for (int sub_k = k; sub_k < max_sub_k; ++sub_k)
                                    val += fourDimRead(Q, b, h, sub_i, sub_k, H, N, d) *
                                        fourDimRead(K, b, h, sub_j, sub_k, H, N, d);
                                twoDimWrite(QK_t, sub_i, sub_j, N, val);
                            }
                        }
#else
                        blockMatmul_T(Q.data() + step, K.data() + step, QK_t.data(),
                                N, d, N, i, k, j, max_sub_i, max_sub_k, max_sub_j);
#endif
                    }
                }
            }

            // 计算softmax
#ifndef ISPC
            for (int i = 0; i < N; ++i) {
                float max_val = 0;
                for (int j = 0; j < N; ++j)
                    max_val = std::max(max_val, twoDimRead(QK_t, i, j, N));

                float l = 0;
                for (int j = 0; j < N; ++j) {
                    float val = exp(twoDimRead(QK_t, i, j, N) - max_val);
                    twoDimWrite(QK_t, i, j, N, val);
                    l += val;
                }

                for (int j = 0; j < N; ++j) {
                    float val = twoDimRead(QK_t, i, j, N) / l;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }
#else
            safeSoftmax(QK_t.data(), N, N);
#endif

            // 分块计算QK^t * V[b][h]
            for (int i = 0; i < N; i += L) {
                for (int j = 0; j < d; j += L) {
                    for (int k = 0; k < N; k += L) {
                        int max_sub_i = std::min(i + L, N);
                        int max_sub_j = std::min(j + L, d);
                        int max_sub_k = std::min(k + L, N);
#ifndef ISPC
                        for (int sub_i = i; sub_i < max_sub_i; ++sub_i) {
                            for (int sub_j = j; sub_j < max_sub_j; ++sub_j) {
                                float val = fourDimRead(O, b, h, sub_i, sub_j, H, N, d);
                                for (int sub_k = k; sub_k < max_sub_k; ++sub_k)
                                    val += twoDimRead(QK_t, sub_i, sub_k, N) *
                                        fourDimRead(V, b, h, sub_k, sub_j, H, N, d);
                                fourDimWrite(O, b, h, sub_i, sub_j, H, N, d, val);
                            }
                        }
#else
                        blockMatmul(QK_t.data(), V.data() + step, O.data() + step,
                                N, N, d, i, k, j, max_sub_i, max_sub_k, max_sub_j);
#endif
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
//    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){
        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){
		        // YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
                std::vector<float> ORow = formatTensor(ORowTensor);
		        //YOUR CODE HERE
		        // 计算ORow
#ifndef ISPC
		        for (int j = 0; j < N; j++) {
		            float val = 0;
		            for (int k = 0; k < d; k++)
		                val += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
		            ORow[j] = val;
		        }
#else
                int step = b * H * N * d + h * N * d;
                matmul_TRow(Q.data() + step + i * d, K.data() + step, ORow.data(), N, d, N, i);
#endif

		        // 计算softmax
#ifndef ISPC
		        float max_val = 0;
                for (int j = 0; j < N; ++j)
                    max_val = std::max(max_val, ORow[j]);
                float l = 0;
                for (int j = 0; j < N; ++j) {
                    float val = exp(ORow[j] - max_val);
                    ORow[j] = val;
                    l += val;
                }
                for (int j = 0; j < N; ++j)
                    ORow[j] /= l;
#else
                safeVecSoftmax(ORow.data(), N);
#endif

                // 计算O[b][h][i]
#ifndef ISPC
                for (int j = 0; j < d; j++) {
		            float val = 0;
		            for (int k = 0; k < N; k++)
		                val += ORow[k] * fourDimRead(V, b, h, k, j, H, N, d);
		            fourDimWrite(O, b, h, i, j, H, N, d, val);
		        }
#else
                matmulRow(ORow.data(), V.data() + step, O.data() + step + i * d, N, N, d, i);
#endif
            }
	    }
    }

	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            std::fill(l.begin(), l.end(), 0);
            for (int j = 0; j < N; j += Bc) {
                // 读入Kj和Vj
                int max_sub_j = std::min(Bc, N - j);
                for (int sub_j = 0; sub_j < max_sub_j; ++sub_j) {
                    for (int k = 0; k < d; ++k) {
                        twoDimWrite(Kj, sub_j, k, d,
                            fourDimRead(K, b, h, sub_j + j, k, H, N, d));
                        twoDimWrite(Vj, sub_j, k, d,
                            fourDimRead(V, b, h, sub_j + j, k, H, N, d));
                    }
                }

                for (int i = 0; i < N; i += Br) {
                    // 读入Qi、Oi、li
                    int max_sub_i = std::min(Br, N - i);
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i) {
                        li[sub_i] = l[i + sub_i];
                        for (int k = 0; k < d; ++k) {
                            twoDimWrite(Qi, sub_i, k, d,
                                fourDimRead(Q, b, h, sub_i + i, k, H, N, d));
                            twoDimWrite(Oi, sub_i, k, d,
                                fourDimRead(O, b, h, sub_i + i, k, H, N, d));
                        }
                    }

                    // 计算Sij = Qi Kj^T
#ifndef ISPC
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i) {
                        for (int sub_j = 0; sub_j < max_sub_j; ++sub_j) {
                            float val = 0;
                            for (int k = 0; k < d; ++k)
                                val += twoDimRead(Qi, sub_i, k, d) * twoDimRead(Kj, sub_j, k, d);
                            twoDimWrite(Sij, sub_i, sub_j, max_sub_j, val);
                        }
                    }
#else
                    matmul_T(Qi.data(), Kj.data(), Sij.data(), max_sub_i, d, max_sub_j);
#endif

                    // 计算Pij = exp(Sij)
#ifndef ISPC
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i)
                        for (int sub_j = 0; sub_j < max_sub_j; ++sub_j)
                            twoDimWrite(Pij, sub_i, sub_j, max_sub_j,
                                exp(twoDimRead(Sij, sub_i, sub_j, max_sub_j)));
#else
                    matexp(Sij.data(), Pij.data(), max_sub_i, max_sub_j);
#endif

                    // 计算lij = rowsum(Pij)
#ifndef ISPC
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i) {
                        float val = 0;
                        for (int sub_j = 0; sub_j < max_sub_j; ++sub_j)
                            val += twoDimRead(Pij, sub_i, sub_j, max_sub_j);
                        lij[sub_i] = val;
                    }
#else
                    rowsum(Pij.data(), lij.data(), max_sub_i, max_sub_j);
#endif

                    // 计算lnew = li + lij
#ifndef ISPC
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i)
                        lnew[sub_i] = li[sub_i] + lij[sub_i];
#else
                    vecsum(li.data(), lij.data(), lnew.data(), max_sub_i);
#endif

                    // 计算Oi = (li * Oi + Pij Vj) / lnew
#ifndef ISPC
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i) {
                        for (int k = 0; k < d; ++k) {
                            float val = 0;
                            for (int sub_j = 0; sub_j < max_sub_j; ++sub_j)
                                val += twoDimRead(Pij, sub_i, sub_j, max_sub_j) * twoDimRead(Vj, sub_j, k, d);
                            val += li[sub_i] * twoDimRead(Oi, sub_i, k, d);
                            val /= lnew[sub_i];
                            twoDimWrite(Oi, sub_i, k, d, val);
                        }
                    }
#else
                    flashAttentionHelper(Pij.data(), Vj.data(), Oi.data(),
                                        li.data(), lnew.data(),
                                        max_sub_i, max_sub_j, d);
#endif

                    // Oi和lnew写回O和l
                    for (int sub_i = 0; sub_i < max_sub_i; ++sub_i) {
                        l[i + sub_i] = lnew[sub_i];
                        for (int k = 0; k < d; ++k)
                            fourDimWrite(O, b, h, sub_i + i, k, H, N, d,
                                twoDimRead(Oi, sub_i, k, d));
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
