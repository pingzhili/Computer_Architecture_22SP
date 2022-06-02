#pragma G++ optimize(0)
#include <cuda.h>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <ctime>
using namespace std;

#define N (1 << 12)
#define BLOCK (1 << 3)

__global__ void gemm_cuda_block(float *A, float *B, float *C);
void gemm_verify(float *A, float *B, float *C);
float random_float();

int main(void) {
    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> A(N*N);
    std::vector<float> B(N*N);
    std::vector<float> C(N*N, 0.0);
    std::generate(A.begin(), A.end(), random_float);
    std::generate(B.begin(), B.end(), random_float);
    dim3 block_size(BLOCK, BLOCK);
    dim3 grid_size((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

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
    float gpu_time;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start, 0);

    gemm_cuda_block <<<grid_size, block_size, 2*BLOCK*BLOCK*sizeof(float)>>> (A_cuda, B_cuda, C_cuda);
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


__global__ void gemm_cuda_block(float* A, float * B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float temp = 0;
    if(i < (N - N%BLOCK) && j < (N - N%BLOCK)) {
        __shared__ float A_sub[BLOCK][BLOCK];
        __shared__ float B_sub[BLOCK][BLOCK];
        for(int k=0; k<N/blockDim.x; k++) {
            A_sub[threadIdx.x%BLOCK][threadIdx.y%BLOCK] = A[blockIdx.x*BLOCK*N + threadIdx.x*N + k*BLOCK + threadIdx.y];
            B_sub[threadIdx.x%BLOCK][threadIdx.y%BLOCK] = B[blockIdx.y*BLOCK + threadIdx.x*N + k*BLOCK*N + threadIdx.y];
            __syncthreads();
            for(int l=0; l<blockDim.x; l++) {
                temp += A_sub[threadIdx.x][l] * B_sub[l][threadIdx.y];
            }
            __syncthreads();
        }
        C[i*N+j] = temp;
    }else if(i < N && j < N){
        for(int k=0; k<N; k++){
            temp += A[i * N + k] * B[k * N + j];
        }
        C[i*N+j] = temp;
    }
}