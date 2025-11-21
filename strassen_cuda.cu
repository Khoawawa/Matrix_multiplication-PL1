// strassen_cuda.cu
// Compile with: nvcc -O3 -arch=sm_70 strassen_cuda.cu -o strassen_cuda
// Run: ./strassen_cuda N

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

using namespace std;

// -------------------- Device kernels --------------------
// elementwise add: C = A + B
__global__ void matAddKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        C[r*strideC + c] = A[r*strideA + c] + B[r*strideB + c];
    }
}

// elementwise sub: C = A - B
__global__ void matSubKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        C[r*strideC + c] = A[r*strideA + c] - B[r*strideB + c];
    }
}

// naive multiply kernel (no shared memory tiling, simple version)
// C = A * B, where A,B,C are n x n with strides
__global__ void matMulNaiveKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        const double* arow = A + row * strideA;
        for (int k = 0; k < n; ++k) {
            sum += arow[k] * B[k * strideB + col];
        }
        C[row * strideC + col] = sum;
    }
}

// -------------------- Host helpers --------------------

struct DevMatrix {
    double* dptr;   // device pointer to start of the matrix block (row-major)
    int n;          // logical block size (n x n)
    int stride;     // actual leading dimension (distance in elements between consecutive rows)
};

// Allocate device matrix with leading dimension = n (contiguous n x n)
double* devAlloc(int n) {
    double* d;
    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;
    CHECK_CUDA(cudaMalloc((void**)&d, bytes));
    CHECK_CUDA(cudaMemset(d, 0, bytes));
    return d;
}

// Host <-> device helpers (host matrices are contiguous row-major n x n)
void hostToDevice(const double* h, double* d, int n) {
    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;
    CHECK_CUDA(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
}
void deviceToHost(const double* d, double* h, int n) {
    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;
    CHECK_CUDA(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
}

// Helper to compute grid/block dims
inline dim3 block2d() { return dim3(16, 16); }
inline dim3 grid2d(int n) { 
    dim3 b = block2d();
    return dim3((n + b.x - 1) / b.x, (n + b.y - 1) / b.y);
}

// Wrapper kernels: add, sub, mul (launched on default stream)
void gpuAdd(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    dim3 g = grid2d(C.n);
    dim3 b = block2d();
    matAddKernel<<<g, b>>>(A.dptr, B.dptr, C.dptr, C.n, A.stride, B.stride, C.stride);
    CHECK_CUDA(cudaGetLastError());
}

void gpuSub(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    dim3 g = grid2d(C.n);
    dim3 b = block2d();
    matSubKernel<<<g, b>>>(A.dptr, B.dptr, C.dptr, C.n, A.stride, B.stride, C.stride);
    CHECK_CUDA(cudaGetLastError());
}

void gpuMulNaive(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    dim3 g = grid2d(C.n);
    dim3 b = block2d();
    matMulNaiveKernel<<<g, b>>>(A.dptr, B.dptr, C.dptr, C.n, A.stride, B.stride, C.stride);
    CHECK_CUDA(cudaGetLastError());
}

// Utility: allocate device scratch n x n, returns DevMatrix with stride = n
DevMatrix devAllocMat(int n) {
    DevMatrix M;
    M.n = n;
    M.stride = n;
    M.dptr = devAlloc(n);
    return M;
}

// Free device matrix
void devFreeMat(DevMatrix& M) {
    if (M.dptr) {
        CHECK_CUDA(cudaFree(M.dptr));
        M.dptr = nullptr;
    }
}

// Copy submatrix device pointers (view) : returns DevMatrix view into base pointer with offset
DevMatrix devView(double* base, int offset_r, int offset_c, int block_n, int stride) {
    DevMatrix v;
    v.n = block_n;
    v.stride = stride;
    v.dptr = base + offset_r * stride + offset_c;
    return v;
}

// Next power of two pad size
int nextPow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Pad host matrix (n x n) into a contiguous host array of size pad x pad (row-major)
vector<double> padHostMatrix(const double* h, int n, int pad) {
    vector<double> out((size_t)pad * (size_t)pad, 0.0);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            out[r*pad + c] = h[r*n + c];
    return out;
}

// Unpad device result (pad x pad) into host (n x n)
void unpadDeviceToHost(const DevMatrix& paddedC, double* hostC, int origN) {
    // copy entire padded matrix to host temp then extract
    vector<double> tmp((size_t)paddedC.n * (size_t)paddedC.n);
    deviceToHost(paddedC.dptr, tmp.data(), paddedC.n);
    for (int r = 0; r < origN; ++r)
        for (int c = 0; c < origN; ++c)
            hostC[r*origN + c] = tmp[r*paddedC.n + c];
}

// -------------------- Strassen (host recursion, GPU ops) --------------------

// threshold: below or equal this we'll use naive GPU multiply
const int BASE_THRESHOLD = 64;

// Strassen recursive on host; A,B,C are DevMatrix blocks (views) on device memory.
// IMPORTANT: The function will allocate/free temp device matrices for intermediate computations.
void strassen_gpu(const DevMatrix& A, const DevMatrix& B, DevMatrix C) {
    int n = A.n;
    if (n <= BASE_THRESHOLD) {
        gpuMulNaive(A, B, C);
        CHECK_CUDA(cudaDeviceSynchronize());
        return;
    }
    int half = n / 2;

    // Sub-block views (assuming A.stride and B.stride >= n, and A.dptr points to top-left of block)
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

    // Allocate temporaries for M1..M7 and some T temps
    DevMatrix M1 = devAllocMat(half);
    DevMatrix M2 = devAllocMat(half);
    DevMatrix M3 = devAllocMat(half);
    DevMatrix M4 = devAllocMat(half);
    DevMatrix M5 = devAllocMat(half);
    DevMatrix M6 = devAllocMat(half);
    DevMatrix M7 = devAllocMat(half);

    DevMatrix T1 = devAllocMat(half);
    DevMatrix T2 = devAllocMat(half);

    // M1 = (a + d) * (e + h)
    gpuAdd(a, d, T1);                // T1 = a + d
    gpuAdd(e, h, T2);                // T2 = e + h
    strassen_gpu(T1, T2, M1);

    // M2 = (c + d) * e
    gpuAdd(c, d, T1);                // T1 = c + d
    strassen_gpu(T1, e, M2);

    // M3 = a * (f - h)
    gpuSub(f, h, T2);                // T2 = f - h
    strassen_gpu(a, T2, M3);

    // M4 = d * (g - e)
    gpuSub(g, e, T1);                // T1 = g - e
    strassen_gpu(d, T1, M4);

    // M5 = (a + b) * h
    gpuAdd(a, b, T1);                // T1 = a + b
    strassen_gpu(T1, h, M5);

    // M6 = (c - a) * (e + f)
    gpuSub(c, a, T1);                // T1 = c - a
    gpuAdd(e, f, T2);                // T2 = e + f
    strassen_gpu(T1, T2, M6);

    // M7 = (b - d) * (g + h)
    gpuSub(b, d, T1);                // T1 = b - d
    gpuAdd(g, h, T2);                // T2 = g + h
    strassen_gpu(T1, T2, M7);

    // Now compute C subblocks:
    // C11 = M1 + M4 - M5 + M7
    // Use T temps to assemble
    gpuAdd(M1, M4, T1);      // T1 = M1 + M4
    gpuSub(T1, M5, T2);      // T2 = T1 - M5
    gpuAdd(T2, M7, c11);     // c11 = T2 + M7

    // C12 = M3 + M5
    gpuAdd(M3, M5, c12);

    // C21 = M2 + M4
    gpuAdd(M2, M4, c21);

    // C22 = M1 + M3 - M2 + M6
    gpuAdd(M1, M3, T1);      // T1 = M1 + M3
    gpuSub(T1, M2, T2);      // T2 = T1 - M2
    gpuAdd(T2, M6, c22);     // c22 = T2 + M6

    // ensure everything finished before freeing
    CHECK_CUDA(cudaDeviceSynchronize());

    // free temps
    devFreeMat(M1); devFreeMat(M2); devFreeMat(M3); devFreeMat(M4);
    devFreeMat(M5); devFreeMat(M6); devFreeMat(M7);
    devFreeMat(T1); devFreeMat(T2);
}

// -------------------- Main / test harness --------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s N\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    if (N <= 0) return 1;

    // random host matrices A and B
    vector<double> hA((size_t)N*(size_t)N), hB((size_t)N*(size_t)N), hC((size_t)N*(size_t)N, 0.0);
    srand(0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            hA[i*N + j] = (double)(rand() % 10);
            hB[i*N + j] = (double)(rand() % 10);
        }

    int P = nextPow2(N); // pad to power of two
    if (P < BASE_THRESHOLD) P = BASE_THRESHOLD; // ensure reasonable base size

    auto paddedA = padHostMatrix(hA.data(), N, P);
    auto paddedB = padHostMatrix(hB.data(), N, P);

    // allocate device padded matrices
    double* dA = nullptr; double* dB = nullptr; double* dC = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(double)*(size_t)P*(size_t)P));
    CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(double)*(size_t)P*(size_t)P));
    CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(double)*(size_t)P*(size_t)P));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(double)*(size_t)P*(size_t)P));

    // host -> device
    CHECK_CUDA(cudaMemcpy(dA, paddedA.data(), sizeof(double)*(size_t)P*(size_t)P, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, paddedB.data(), sizeof(double)*(size_t)P*(size_t)P, cudaMemcpyHostToDevice));

    DevMatrix Adev{dA, P, P}, Bdev{dB, P, P}, Cdev{dC, P, P};

    printf("Running Strassen on GPU (N=%d, padded=%d)\n", N, P);

    auto t0 = chrono::high_resolution_clock::now();
    strassen_gpu(Adev, Bdev, Cdev);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = chrono::high_resolution_clock::now();

    double elapsed = chrono::duration<double>(t1 - t0).count();
    printf("Strassen time (GPU): %g s\n", elapsed);

    // copy result back and unpad
    unpadDeviceToHost(Cdev, hC.data(), N);

    // Optionally verify against CPU naive multiplication for small N
    if (N <= 256) {
        vector<double> ref((size_t)N*(size_t)N, 0.0);
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j)
                    ref[i*N + j] += hA[i*N + k] * hB[k*N + j];

        bool ok = true;
        for (int i = 0; i < N && ok; ++i)
            for (int j = 0; j < N; ++j)
                if (fabs(ref[i*N + j] - hC[i*N + j]) > 1e-6) {
                    printf("Mismatch at (%d,%d): ref=%g got=%g\n", i, j, ref[i*N + j], hC[i*N + j]);
                    ok = false; break;
                }
        printf("Verification: %s\n", ok ? "OK" : "FAIL");
    }

    // free device memory
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    return 0;
}
