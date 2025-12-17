#include <omp.h>
#include <bits/stdc++.h>
#include "Matrix.h" // Vẫn cần include file này để dùng class Matrix/MatrixView

using namespace std;

// --- Helper Functions ---
auto saveMatrixToFile = [](const Matrix<double>& M, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return;
    int n = M.get_n();
    fout << std::setprecision(10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << M.get(i, j) << (j != n - 1 ? " " : "");
        }
        fout << "\n";
    }
    fout.close();
};

template<typename T>
bool compareMatrices(const Matrix<T>& A, const Matrix<T>& B, double epsilon = 1e-6) {
    if (A.get_n() != B.get_n()) return false;
    int n = A.get_n();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(A.get(i, j) - B.get(i, j)) > epsilon) return false;
        }
    }
    return true; 
}

// --- Arithmetic Operations ---
template <typename T> void add(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> C) {
    #pragma omp parallel for
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) C.at(i, j) = A.at(i, j) + B.at(i, j);
    }
}

template <typename T> void sub(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> C) {
    #pragma omp parallel for
    for (int i = 0; i < A.n; i++) {
        for (int j = 0; j < A.n; j++) C.at(i, j) = A.at(i, j) - B.at(i, j);
    }
}

template <typename T> void naiveMultiplyMP(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C) {
    // Tối ưu loop i-k-j để tận dụng Cache
    #pragma omp parallel for
    for (int i = 0; i < A.n; i++) {
        for (int k = 0; k < A.n; k++) {
            T r = A.at(i, k);
            for (int j = 0; j < A.n; j++) {
                C.at(i, j) += r * B.at(k, j);
            }
        }
    }
}

template <typename T> 
void tiledMultiplyMP(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C) {
    int n = A.n;
    int BLOCK_SIZE = 64; // Kích thước block tối ưu cho L1/L2 Cache

    // Reset C về 0 trước khi cộng dồn (quan trọng!)
    #pragma omp parallel for collapse(2)
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) C.at(i,j) = 0;
    }

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
            // Loop j nằm trong để tối ưu vector hóa
            for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
                
                int i_end = min(i0 + BLOCK_SIZE, n);
                int k_end = min(k0 + BLOCK_SIZE, n);
                int j_end = min(j0 + BLOCK_SIZE, n);
                
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        T r = A.at(i, k);
                        for (int j = j0; j < j_end; j++) {
                            C.at(i, j) += r * B.at(k, j);
                        }
                    }
                }
            }
        }
    }
}

// --- Recursive Strassen with DYNAMIC PADDING ---

template <typename T> 
void strassenRecursive(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C) {
    int n = A.n;

    // Base case: Dùng Tiled Multiply thay vì nhân thường
    // Threshold 512 là điểm ngọt giữa chi phí tạo task và hiệu năng tính toán
    if (n <= 512) { 
        tiledMultiplyMP(A, B, C);
        return;
    }

    // Dynamic Padding: Nếu LẺ thì đệm +1 (thay vì lên lũy thừa 2)
    if (n % 2 != 0) {
        int m = n + 1;
        Matrix<T> APad(m), BPad(m), CPad(m);
        MatrixView<T> vA = APad.view(), vB = BPad.view(), vC = CPad.view();

        // Copy A, B sang pad (Parallel)
        #pragma omp parallel for collapse(2)
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                vA.at(i,j) = A.at(i,j);
                vB.at(i,j) = B.at(i,j);
            }
        }
        
        strassenRecursive(vA, vB, vC); // Đệ quy với kích thước chẵn (n+1)

        // Copy kết quả về C
        #pragma omp parallel for collapse(2)
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                C.at(i,j) = vC.at(i,j);
            }
        }
        return;
    }

    // Logic Strassen (n chẵn)
    int half = n / 2;
    MatrixView<T> a{A.data, half, A.stride};
    MatrixView<T> b{A.data + half, half, A.stride};
    MatrixView<T> c{A.data + half * A.stride, half, A.stride};
    MatrixView<T> d{A.data + half * A.stride + half, half, A.stride};
    
    MatrixView<T> e{B.data, half, B.stride};
    MatrixView<T> f{B.data + half, half, B.stride};
    MatrixView<T> g{B.data + half * B.stride, half, B.stride};
    MatrixView<T> h{B.data + half * B.stride + half, half, B.stride};

    MatrixView<T> c11{C.data, half, C.stride};
    MatrixView<T> c12{C.data + half, half, C.stride};
    MatrixView<T> c21{C.data + half * C.stride, half, C.stride};
    MatrixView<T> c22{C.data + half * C.stride + half, half, C.stride};

    Matrix<T> m1(half), m2(half), m3(half), m4(half), m5(half), m6(half), m7(half);
    MatrixView<T> vm1=m1.view(), vm2=m2.view(), vm3=m3.view(), vm4=m4.view(), vm5=m5.view(), vm6=m6.view(), vm7=m7.view();

    #pragma omp task shared(vm1)
    {
        Matrix<T> t1(half), t2(half);
        add(a, d, t1.view()); add(e, h, t2.view());
        strassenRecursive(t1.view(), t2.view(), vm1);
    }
    #pragma omp task shared(vm2)
    {
        Matrix<T> t1(half);
        add(c, d, t1.view());
        strassenRecursive(t1.view(), e, vm2);
    }
    #pragma omp task shared(vm3)
    {
        Matrix<T> t1(half);
        sub(f, h, t1.view());
        strassenRecursive(a, t1.view(), vm3);
    }
    #pragma omp task shared(vm4)
    {
        Matrix<T> t1(half);
        sub(g, e, t1.view());
        strassenRecursive(d, t1.view(), vm4);
    }
    #pragma omp task shared(vm5)
    {
        Matrix<T> t1(half);
        add(a, b, t1.view());
        strassenRecursive(t1.view(), h, vm5);
    }
    #pragma omp task shared(vm6)
    {
        Matrix<T> t1(half), t2(half);
        sub(c, a, t1.view()); add(e, f, t2.view());
        strassenRecursive(t1.view(), t2.view(), vm6);
    }
    #pragma omp task shared(vm7)
    {
        Matrix<T> t1(half), t2(half);
        sub(b, d, t1.view()); add(g, h, t2.view());
        strassenRecursive(t1.view(), t2.view(), vm7);
    }
    #pragma omp taskwait

    #pragma omp parallel sections
    {
        #pragma omp section 
        { add(vm1, vm4, c11); sub(c11, vm5, c11); add(c11, vm7, c11); }
        #pragma omp section 
        { add(vm3, vm5, c12); }
        #pragma omp section 
        { add(vm2, vm4, c21); }
        #pragma omp section 
        { sub(vm1, vm2, c22); add(c22, vm3, c22); add(c22, vm6, c22); }
    }
}

// Wrapper Interface
template <typename T>
void strassenWithPadding(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            strassenRecursive(A, B, C);
        }
    }
}