#pragma G++ optimize(0)
#include <cuda.h>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
using namespace std;

#define N (1 << 10)
#define GRID_SIZE (1 << 7)
#define BLOCK_SIZE (1 << 6)

__global__ void gemm_baseline(float *A, float *B, float *C);
void gemm_verify(float *A, float *B, float *C);
float random_float();

int main(void) {
    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> A(N*N);
    std::vector<float> B(N*N);
    std::vector<float> C(N*N, 0.0);
    std::generate(A.begin(), A.end(), random_float);
    std::generate(B.begin(), B.end(), random_float);
    dim3 grid_size(GRID_SIZE, GRID_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);


    float *A_cuda, *B_cuda, *C_cuda;
    if (cudaMalloc((void **)&A_cuda, N*N*sizeof(float)) != cudaSuccess) {
        cout << "cudaMalloc failed" << endl;
        return -1;
    }
    if (cudaMalloc((void **)&B_cuda, N*N*sizeof(float)) != cudaSuccess) {
        cout << "cudaMalloc failed" << endl;
        return -1;
    }
    if (cudaMalloc((void **)&C_cuda, N*N*sizeof(float)) != cudaSuccess) {
        cout << "cudaMalloc failed" << endl;
        return -1;
    }
    if (cudaMemcpy(A_cuda, A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cout << "cudaMemcpy failed" << endl;
        return -1;
    }
    if (cudaMemcpy(B_cuda, B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cout << "cudaMemcpy failed" << endl;
        return -1;
    }

    cudaEvent_t cuda_start, cuda_stop;
    float gpu_time = 0.0;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start, 0);

    gemm_baseline <<<grid_size, block_size>>> (A_cuda, B_cuda, C_cuda);
    cudaDeviceSynchronize();

    cudaEventRecord(cuda_stop, 0);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    if (cudaMemcpy(C.data(), C_cuda, N*N*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cout << "result cudaMemcpy failed" << endl;
        return -1;
    }
    cudaFree(A_cuda);
    cudaFree(B_cuda);
    cudaFree(C_cuda);

    gemm_verify(A.data(), B.data(), C.data());
    cout << "time: " << gpu_time << endl;
    return 0;
}

float random_float(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
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


__global__ void gemm_baseline(float* A, float * B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float temp = 0;
    if((i < N) && (j < N)){
        for(int k=0; k<N; k++){
            temp += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = temp;
    }
}