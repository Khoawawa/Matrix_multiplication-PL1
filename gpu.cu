// gpu.cu
// Compile: nvcc -O3 -arch=sm_70 gpu.cu -o gpu
// Run: ./gpu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

using namespace std;

// -------------------- Device kernels --------------------
__global__ void matAddKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        C[r*strideC + c] = A[r*strideA + c] + B[r*strideB + c];
    }
}

__global__ void matSubKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        C[r*strideC + c] = A[r*strideA + c] - B[r*strideB + c];
    }
}

__global__ void matMulNaiveKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        // Duyệt k
        for (int k = 0; k < n; ++k) {
            sum += A[row * strideA + k] * B[k * strideB + col];
        }
        C[row * strideC + col] = sum;
    }
}

// -------------------- Host helpers --------------------

struct DevMatrix {
    double* dptr;   
    int n;          
    int stride;     
};

double* devAlloc(int n) {
    double* d;
    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;
    CHECK_CUDA(cudaMalloc((void**)&d, bytes));
    CHECK_CUDA(cudaMemset(d, 0, bytes));
    return d;
}

inline dim3 block2d() { return dim3(16, 16); }
inline dim3 grid2d(int n) { 
    dim3 b = block2d();
    return dim3((n + b.x - 1) / b.x, (n + b.y - 1) / b.y);
}

void gpuAdd(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    dim3 g = grid2d(C.n);
    dim3 b = block2d();
    matAddKernel<<<g, b>>>(A.dptr, B.dptr, C.dptr, C.n, A.stride, B.stride, C.stride);
}

void gpuSub(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    dim3 g = grid2d(C.n);
    dim3 b = block2d();
    matSubKernel<<<g, b>>>(A.dptr, B.dptr, C.dptr, C.n, A.stride, B.stride, C.stride);
}

void gpuMulNaive(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    dim3 g = grid2d(C.n);
    dim3 b = block2d();
    matMulNaiveKernel<<<g, b>>>(A.dptr, B.dptr, C.dptr, C.n, A.stride, B.stride, C.stride);
}

DevMatrix devAllocMat(int n) {
    DevMatrix M;
    M.n = n;
    M.stride = n;
    M.dptr = devAlloc(n);
    return M;
}

void devFreeMat(DevMatrix& M) {
    if (M.dptr) {
        CHECK_CUDA(cudaFree(M.dptr));
        M.dptr = nullptr;
    }
}

DevMatrix devView(double* base, int offset_r, int offset_c, int block_n, int stride) {
    DevMatrix v;
    v.n = block_n;
    v.stride = stride;
    v.dptr = base + offset_r * stride + offset_c;
    return v;
}

int nextPow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

void hostToDevicePad(const double* h, double* d, int n, int pad) {
    vector<double> tempRow(pad, 0.0);
    for(int i = 0; i < n; i++) {
        memcpy(tempRow.data(), h + i * n, n * sizeof(double));
        CHECK_CUDA(cudaMemcpy(d + i * pad, tempRow.data(), pad * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void deviceToHostUnpad(const double* d, double* h, int n, int pad) {
    vector<double> tempRow(pad);
    for(int i = 0; i < n; i++) {
        CHECK_CUDA(cudaMemcpy(tempRow.data(), d + i * pad, pad * sizeof(double), cudaMemcpyDeviceToHost));
        memcpy(h + i * n, tempRow.data(), n * sizeof(double));
    }
}

// -------------------- Strassen Logic --------------------

const int BASE_THRESHOLD = 64; 

void strassen_gpu(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    int n = A.n;
    if (n <= BASE_THRESHOLD) {
        gpuMulNaive(A, B, C);
        return; 
    }
    int half = n / 2;

    DevMatrix a = devView(A.dptr, 0, 0, half, A.stride);
    DevMatrix b = devView(A.dptr, 0, half, half, A.stride);
    DevMatrix c = devView(A.dptr, half, 0, half, A.stride);
    DevMatrix d = devView(A.dptr, half, half, half, A.stride);

    DevMatrix e = devView(B.dptr, 0, 0, half, B.stride);
    DevMatrix f = devView(B.dptr, 0, half, half, B.stride);
    DevMatrix g = devView(B.dptr, half, 0, half, B.stride);
    DevMatrix h = devView(B.dptr, half, half, half, B.stride);

    DevMatrix c11 = devView(C.dptr, 0, 0, half, C.stride);
    DevMatrix c12 = devView(C.dptr, 0, half, half, C.stride);
    DevMatrix c21 = devView(C.dptr, half, 0, half, C.stride);
    DevMatrix c22 = devView(C.dptr, half, half, half, C.stride);

    DevMatrix M1 = devAllocMat(half);
    DevMatrix M2 = devAllocMat(half);
    DevMatrix M3 = devAllocMat(half);
    DevMatrix M4 = devAllocMat(half);
    DevMatrix M5 = devAllocMat(half);
    DevMatrix M6 = devAllocMat(half);
    DevMatrix M7 = devAllocMat(half);
    DevMatrix T1 = devAllocMat(half);
    DevMatrix T2 = devAllocMat(half);

    gpuAdd(a, d, T1); gpuAdd(e, h, T2); strassen_gpu(T1, T2, M1);
    gpuAdd(c, d, T1); strassen_gpu(T1, e, M2);
    gpuSub(f, h, T2); strassen_gpu(a, T2, M3);
    gpuSub(g, e, T1); strassen_gpu(d, T1, M4);
    gpuAdd(a, b, T1); strassen_gpu(T1, h, M5);
    gpuSub(c, a, T1); gpuAdd(e, f, T2); strassen_gpu(T1, T2, M6);
    gpuSub(b, d, T1); gpuAdd(g, h, T2); strassen_gpu(T1, T2, M7);

    gpuAdd(M1, M4, T1); gpuSub(T1, M5, T2); gpuAdd(T2, M7, c11);
    gpuAdd(M3, M5, c12);
    gpuAdd(M2, M4, c21);
    gpuAdd(M1, M3, T1); gpuSub(T1, M2, T2); gpuAdd(T2, M6, c22);

    CHECK_CUDA(cudaDeviceSynchronize()); 

    devFreeMat(M1); devFreeMat(M2); devFreeMat(M3); devFreeMat(M4);
    devFreeMat(M5); devFreeMat(M6); devFreeMat(M7);
    devFreeMat(T1); devFreeMat(T2);
}

// -------------------- TEST SUITE --------------------

bool checkEqual(const vector<double>& A, const vector<double>& B, int n) {
    for(size_t i = 0; i < (size_t)n*n; ++i) {
        if (fabs(A[i] - B[i]) > 1e-4) {
            printf("  -> Mismatch at index %lu: Ref %g != Strassen %g\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void runBenchmark(int N) {
    cout << "--------------------------------------------------------" << endl;
    cout << "Testing Size N = " << N << endl;
    
    int P = nextPow2(N);
    if (P < BASE_THRESHOLD) P = BASE_THRESHOLD;
    cout << "  (Padding to P = " << P << " for Strassen)" << endl;

    vector<double> hA(N * N), hB(N * N);
    vector<double> hC_Naive(N * N, 0.0);
    vector<double> hC_Strassen(N * N, 0.0);

    for (int i = 0; i < N * N; ++i) {
        hA[i] = (double)(rand() % 5);
        hB[i] = (double)(rand() % 5);
    }

    {
        double *dA, *dB, *dC;
        size_t sz = sizeof(double) * N * N;
        CHECK_CUDA(cudaMalloc(&dA, sz)); CHECK_CUDA(cudaMalloc(&dB, sz)); CHECK_CUDA(cudaMalloc(&dC, sz));
        CHECK_CUDA(cudaMemcpy(dA, hA.data(), sz, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), sz, cudaMemcpyHostToDevice));

        DevMatrix mA = {dA, N, N};
        DevMatrix mB = {dB, N, N};
        DevMatrix mC = {dC, N, N};

        auto t1 = chrono::high_resolution_clock::now();
        gpuMulNaive(mA, mB, mC);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t2 = chrono::high_resolution_clock::now();
        double time = chrono::duration<double>(t2 - t1).count();
        
        cout << "  [Naive GPU]    Time: " << fixed << setprecision(4) << time << " s" << endl;
        
        CHECK_CUDA(cudaMemcpy(hC_Naive.data(), dC, sz, cudaMemcpyDeviceToHost));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    {
        double *sdA, *sdB, *sdC;
        size_t padSz = sizeof(double) * P * P;
        CHECK_CUDA(cudaMalloc(&sdA, padSz)); CHECK_CUDA(cudaMalloc(&sdB, padSz)); CHECK_CUDA(cudaMalloc(&sdC, padSz));
        CHECK_CUDA(cudaMemset(sdC, 0, padSz));

        hostToDevicePad(hA.data(), sdA, N, P);
        hostToDevicePad(hB.data(), sdB, N, P);

        DevMatrix smA = {sdA, P, P};
        DevMatrix smB = {sdB, P, P};
        DevMatrix smC = {sdC, P, P};

        auto t1 = chrono::high_resolution_clock::now();
        strassen_gpu(smA, smB, smC);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t2 = chrono::high_resolution_clock::now();
        double time = chrono::duration<double>(t2 - t1).count();

        cout << "  [Strassen GPU] Time: " << fixed << setprecision(4) << time << " s" << endl;

        deviceToHostUnpad(sdC, hC_Strassen.data(), N, P);
        cudaFree(sdA); cudaFree(sdB); cudaFree(sdC);
    }

    if (checkEqual(hC_Naive, hC_Strassen, N)) {
        cout << "  [Verification] \033[1;32mPASSED\033[0m" << endl;
    } else {
        cout << "  [Verification] \033[1;31mFAILED\033[0m" << endl;
    }
}

int main(int argc, char* argv[]) {
    srand(42);
    
    if (argc > 1) {
        int n = atoi(argv[1]);
        runBenchmark(n);
    } else {
        vector<int> tests = {1024, 2048, 4000, 4096};
        for (int n : tests) {
            runBenchmark(n);
        }
    }
    return 0;
}