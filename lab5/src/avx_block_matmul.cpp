/* g++-11 -mavx2 -march=native -o avx_block_matmul avx_block_matmul.cpp */
#pragma G++ optimize(0)
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
#include <immintrin.h>

#define BLOCK_SIZE 256
#define UNROLL 32
using namespace std;


int N = (1 << 9);
void gemm_verify(float *A, float *B, float *C); // you can use inline function
void gemm_avx_block(float *A, float *B, float *C); // you can use inline function

float random_float(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}


int main(void) {
    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> A(N*N);
    std::vector<float> B(N*N);
    std::vector<float> C(N*N, 0.0);
    std::generate(A.begin(), A.end(), random_float);
    std::generate(B.begin(), B.end(), random_float);
    auto start = clock();
    gemm_avx_block(A.data(), B.data(), C.data());
    auto end = clock();
    gemm_verify(A.data(), B.data(), C.data());
    cout << "time: " << (end - start) / double(CLOCKS_PER_SEC) << endl;
    return 0;
}


void gemm_verify(float *A, float *B, float *C) {
    float temp;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            temp = 0.0;
            for(int k=0; k<N; k++){
                temp += A[k*N+i] * B[j*N+k];
            }
            if(temp-C[j*N+i] > 0.0001 || temp-C[j*N+i] < -0.0001) {
                cout << "Verify failed at " << "(i=" << i << ", " << "j=" << j << ")" << endl;
                cout << temp << endl;
                cout << C[i*N+j] << endl;
                return;
            }
        }
    }
    cout << "Verify success!" << endl;
}


void gemm_avx_block(float *A, float *B, float *C){
    for(int i=0; i<N; i+=BLOCK_SIZE){
        for(int j=0; j<N; j+=BLOCK_SIZE){
            for(int k=0; k<N; k+=BLOCK_SIZE){
                for(int m=i; m<i+BLOCK_SIZE; m+=8*UNROLL){
                    for(int n=j; n<j+BLOCK_SIZE; n++){
                        __m256 c[UNROLL];
                        for(int x=0; x<UNROLL; x++){
                            c[x] = _mm256_load_ps(C+8*x+m+n*N); // c[x] = C[m][n]
                        }
                        for(int p=k; p<k+BLOCK_SIZE; p++){
                            __m256 b = _mm256_broadcast_ss(B+p+n*N); // b = B[p][n]
                            for(int x=0; x<UNROLL; x++){
                                c[x] = _mm256_add_ps(c[x], _mm256_mul_ps(_mm256_load_ps(A+N*p+x*8+m), b)); // A[m][p]
                            }
                        }
                        for(int x=0; x<UNROLL; x++){
                            _mm256_store_ps(C+m+8*x+n*N, c[x]);
                        }
                    }
                }
            }
        }
    }
}
