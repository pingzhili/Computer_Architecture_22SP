#pragma G++ optimize(0)
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
using namespace std;


int N = (1 << 10);

void gemm_baseline(float *A, float *B, float *C); // you can use inline function

float random_float(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);
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
    gemm_baseline(A.data(), B.data(), C.data());
    auto end = clock();
    cout << "time: " << (end - start) / double(CLOCKS_PER_SEC) << endl;
    return 0;
}

void gemm_baseline(float *A, float *B, float *C) {
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
}