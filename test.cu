// Compile: nvcc -O3 -arch=sm_70 main_compare.cu -o compare_gpu
// Run: ./compare_gpu

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

// ==========================================
// 1. CUDA KERNELS
// ==========================================

// Kernel cộng: C = A + B
__global__ void matAddKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        C[r*strideC + c] = A[r*strideA + c] + B[r*strideB + c];
    }
}

// Kernel trừ: C = A - B
__global__ void matSubKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        C[r*strideC + c] = A[r*strideA + c] - B[r*strideB + c];
    }
}

// Kernel Nhân thường (Naive): C = A * B
// Lưu ý: Đây là bản cài đặt đơn giản nhất (Global Memory access), không dùng Shared Memory Tiling.
// Nó sẽ chậm, nhưng đúng nghĩa là "Naive" để so sánh.
__global__ void matMulNaiveKernel(const double* A, const double* B, double* C, int n, int strideA, int strideB, int strideC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        // Duyệt qua k
        for (int k = 0; k < n; ++k) {
            sum += A[row * strideA + k] * B[k * strideB + col];
        }
        C[row * strideC + col] = sum;
    }
}

// ==========================================
// 2. HELPER FUNCTIONS & STRUCTS
// ==========================================

struct DevMatrix {
    double* dptr;   
    int n;          
    int stride;     
};

// Cấu hình Grid/Block
inline dim3 block2d() { return dim3(16, 16); }
inline dim3 grid2d(int n) { 
    dim3 b = block2d();
    return dim3((n + b.x - 1) / b.x, (n + b.y - 1) / b.y);
}

// Cấp phát bộ nhớ GPU
double* devAlloc(int n) {
    double* d;
    size_t bytes = sizeof(double) * n * n;
    CHECK_CUDA(cudaMalloc((void**)&d, bytes));
    CHECK_CUDA(cudaMemset(d, 0, bytes));
    return d;
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

// Tạo View (không cấp phát mới)
DevMatrix devView(double* base, int offset_r, int offset_c, int block_n, int stride) {
    DevMatrix v;
    v.n = block_n;
    v.stride = stride;
    v.dptr = base + offset_r * stride + offset_c;
    return v;
}

// Hàm padding lên lũy thừa 2
int nextPow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Copy từ Host sang Device (có padding nếu cần)
void hostToDevicePad(const double* h, double* d, int n, int pad) {
    // Copy từng dòng để xử lý padding (nếu n != pad)
    vector<double> tempRow(pad, 0.0);
    for(int i = 0; i < n; i++) {
        // Copy data dòng i vào temp
        memcpy(tempRow.data(), h + i * n, n * sizeof(double));
        // Copy temp xuống GPU tại vị trí đúng
        CHECK_CUDA(cudaMemcpy(d + i * pad, tempRow.data(), pad * sizeof(double), cudaMemcpyHostToDevice));
    }
    // Các dòng đệm bên dưới (từ n -> pad) đã được memset 0 lúc alloc
}

// Copy từ Device về Host (Unpad)
void deviceToHostUnpad(const double* d, double* h, int n, int pad) {
    vector<double> tempRow(pad);
    for(int i = 0; i < n; i++) {
        CHECK_CUDA(cudaMemcpy(tempRow.data(), d + i * pad, pad * sizeof(double), cudaMemcpyDeviceToHost));
        memcpy(h + i * n, tempRow.data(), n * sizeof(double));
    }
}

// So sánh kết quả
bool verifyResult(const vector<double>& C1, const vector<double>& C2, int n) {
    for(int i=0; i<n*n; i++) {
        if(fabs(C1[i] - C2[i]) > 1e-4) {
            printf("Mismatch at index %d: %f vs %f\n", i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

// Wrappers gọi Kernel
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

// ==========================================
// 3. STRASSEN IMPLEMENTATION
// ==========================================

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

    // Strassen formulas
    gpuAdd(a, d, T1); gpuAdd(e, h, T2); strassen_gpu(T1, T2, M1); // M1
    gpuAdd(c, d, T1); strassen_gpu(T1, e, M2);                    // M2
    gpuSub(f, h, T2); strassen_gpu(a, T2, M3);                    // M3
    gpuSub(g, e, T1); strassen_gpu(d, T1, M4);                    // M4
    gpuAdd(a, b, T1); strassen_gpu(T1, h, M5);                    // M5
    gpuSub(c, a, T1); gpuAdd(e, f, T2); strassen_gpu(T1, T2, M6); // M6
    gpuSub(b, d, T1); gpuAdd(g, h, T2); strassen_gpu(T1, T2, M7); // M7

    // Assemble C
    gpuAdd(M1, M4, T1); gpuSub(T1, M5, T2); gpuAdd(T2, M7, c11);  // C11
    gpuAdd(M3, M5, c12);                                          // C12
    gpuAdd(M2, M4, c21);                                          // C21
    gpuAdd(M1, M3, T1); gpuSub(T1, M2, T2); gpuAdd(T2, M6, c22);  // C22

    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for children

    devFreeMat(M1); devFreeMat(M2); devFreeMat(M3); devFreeMat(M4);
    devFreeMat(M5); devFreeMat(M6); devFreeMat(M7);
    devFreeMat(T1); devFreeMat(T2);
}

// ==========================================
// 4. MAIN BENCHMARK SUITE
// ==========================================

void runTest(int N) {
    cout << "\n=========================================" << endl;
    cout << ">>> Testing Size: " << N << "x" << N << endl;
    cout << "-----------------------------------------" << endl;

    // 1. Setup Data
    vector<double> hA(N * N);
    vector<double> hB(N * N);
    vector<double> hC_Naive(N * N, 0.0);
    vector<double> hC_Strassen(N * N, 0.0);

    for (int i = 0; i < N * N; i++) {
        hA[i] = (double)(rand() % 10);
        hB[i] = (double)(rand() % 10);
    }

    // 2. Run GPU Naive
    // Với Naive, ta có thể dùng trực tiếp ma trận NxN không cần padding 
    // (nhưng để công bằng về memory layout, ta vẫn alloc)
    {
        double *dA, *dB, *dC;
        size_t size = N * N * sizeof(double);
        CHECK_CUDA(cudaMalloc((void**)&dA, size));
        CHECK_CUDA(cudaMalloc((void**)&dB, size));
        CHECK_CUDA(cudaMalloc((void**)&dC, size));

        CHECK_CUDA(cudaMemcpy(dA, hA.data(), size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dC, 0, size));

        DevMatrix matA = {dA, N, N};
        DevMatrix matB = {dB, N, N};
        DevMatrix matC = {dC, N, N};

        auto t1 = chrono::high_resolution_clock::now();
        gpuMulNaive(matA, matB, matC);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t2 = chrono::high_resolution_clock::now();
        double timeNaive = chrono::duration<double>(t2 - t1).count();

        cout << "GPU Naive Time:    " << fixed << setprecision(5) << timeNaive << " s" << endl;

        CHECK_CUDA(cudaMemcpy(hC_Naive.data(), dC, size, cudaMemcpyDeviceToHost));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);

        // 3. Run GPU Strassen
        int P = nextPow2(N);
        if (P < BASE_THRESHOLD) P = BASE_THRESHOLD;

        double *sdA, *sdB, *sdC;
        size_t padSize = P * P * sizeof(double);
        CHECK_CUDA(cudaMalloc((void**)&sdA, padSize));
        CHECK_CUDA(cudaMalloc((void**)&sdB, padSize));
        CHECK_CUDA(cudaMalloc((void**)&sdC, padSize));
        CHECK_CUDA(cudaMemset(sdC, 0, padSize));

        // Copy có Padding
        hostToDevicePad(hA.data(), sdA, N, P);
        hostToDevicePad(hB.data(), sdB, N, P);

        DevMatrix sMatA = {sdA, P, P};
        DevMatrix sMatB = {sdB, P, P};
        DevMatrix sMatC = {sdC, P, P};

        t1 = chrono::high_resolution_clock::now();
        strassen_gpu(sMatA, sMatB, sMatC);
        CHECK_CUDA(cudaDeviceSynchronize());
        t2 = chrono::high_resolution_clock::now();
        double timeStrassen = chrono::duration<double>(t2 - t1).count();

        cout << "GPU Strassen Time: " << fixed << setprecision(5) << timeStrassen << " s" << endl;

        // Copy về và Unpad
        deviceToHostUnpad(sdC, hC_Strassen.data(), N, P);

        cudaFree(sdA); cudaFree(sdB); cudaFree(sdC);

        // 4. Compare & Report
        bool pass = verifyResult(hC_Naive, hC_Strassen, N);
        
        cout << "-----------------------------------------" << endl;
        if (pass) cout << "Result Verification: \033[1;32mPASSED\033[0m" << endl;
        else      cout << "Result Verification: \033[1;31mFAILED\033[0m" << endl;
        
        if (pass) {
            double speedup = timeNaive / timeStrassen;
            cout << "Speedup (Naive/Strassen): " << speedup << "x" << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    srand(42);
    
    // Test các kích thước khác nhau
    // Lưu ý: Naive O(N^3) sẽ rất chậm khi N lớn.
    // Với N=4096, Naive có thể chạy mất vài giây đến vài chục giây tùy GPU.
    vector<int> sizes = {512, 1024, 2048, 4096}; 

    // Nếu người dùng nhập tham số, chỉ chạy size đó
    if (argc > 1) {
        sizes.clear();
        sizes.push_back(atoi(argv[1]));
    }

    for (int n : sizes) {
        runTest(n);
    }

    return 0;
}