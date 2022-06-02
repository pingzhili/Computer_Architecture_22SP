/* g++-11 -mavx2 -march=native -o avx_matmul avx_matmul.cpp */
#pragma G++ optimize(0)
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
#include <immintrin.h>

using namespace std;


int N = (1 << 10);
void gemm_verify(float *A, float *B, float *C); // you can use inline function
void gemm_avx(float *A, float *B, float *C); // you can use inline function


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
    gemm_avx(A.data(), B.data(), C.data());
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
                temp += A[i*N+k] * B[k*N+j];
            }
            if(temp-C[i*N+j] > 0.0001 || temp-C[i*N+j] < -0.0001) {
                cout << "Verify failed at " << "(" << i << ", " << j << ")" << endl;
                cout << temp << endl;
                cout << C[i*N+j] << endl;
                return;
            }
        }
    }
    cout << "Verify success!" << endl;
}


void gemm_avx(float *A, float *B, float *C) {
    __m256 a, b, c;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j+=8){
            for(int k=0; k<N; k++){
                c = _mm256_load_ps(C+i*N+j);
                a = _mm256_broadcast_ss(A+i*N+k);
                b = _mm256_load_ps(B+k*N+j);
                c = _mm256_fmadd_ps(a, b, c);
                _mm256_store_ps(C+i*N+j, c);
            }
        }
    }
}

