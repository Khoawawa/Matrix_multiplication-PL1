#include <omp.h>
#include <bits/stdc++.h>
#include "Matrix.h"

using namespace std;

auto saveMatrixToFile = [](const Matrix<double>& M, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error: Cannot open " << filename << " for writing\n";
        return;
    }
    int n = M.get_n();
    fout << std::setprecision(10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << M.get(i, j);
            if (j != n - 1) fout << " ";
        }
        fout << "\n";
    }
    fout.close();
    std::cout << "Saved " << filename << " (" << n << "x" << n << ")\n";
};

template<typename T>
bool compareMatrices(const Matrix<T>& A, const Matrix<T>& B, double epsilon = 1e-6)
{
    if (A.get_n() != B.get_n()) {
        return false;
    }
    int n = A.get_n();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(A.get(i, j) - B.get(i, j)) > epsilon) {
                std::cerr << "Verification Failed at (" << i << "," << j << "): " 
                          << A.get(i, j) << " != " << B.get(i, j) << std::endl;
                return false; 
            }
        }
    }
    return true; 
}

template <typename T> void add(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> C)
{
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.n; i++)
    {
        for (int j = 0; j < A.n; j++)
        {
            C.at(i, j) = A.at(i, j) + B.at(i, j);
        }
    }
}

template <typename T> void sub(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> C)
{
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.n; i++)
    {
        for (int j = 0; j < A.n; j++)
        {
            C.at(i, j) = A.at(i, j) - B.at(i, j);
        }
    }
}

template <typename T> void naiveMultiplyMP(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.n; i++)
    {
        for (int j = 0; j < A.n; j++)
        {
            T sum = 0;
            for (int k = 0; k < A.n; k++)
            {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}

template <typename T> void strassenMP(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C) {
    if (A.n <= 128) {
        naiveMultiplyMP(A,B,C);
        return;
    }
    int half = A.n / 2;
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

    Matrix<T> m1(half);
    Matrix<T> m2(half);
    Matrix<T> m3(half);
    Matrix<T> m4(half);
    Matrix<T> m5(half);
    Matrix<T> m6(half);
    Matrix<T> m7(half); 

    MatrixView<T> m1_view = m1.view();
    MatrixView<T> m2_view = m2.view();
    MatrixView<T> m3_view = m3.view();
    MatrixView<T> m4_view = m4.view();
    MatrixView<T> m5_view = m5.view();
    MatrixView<T> m6_view = m6.view();
    MatrixView<T> m7_view = m7.view();

    // M1 = (a+d)(e+h)
    #pragma omp task
    {
        Matrix<T> ad(half); 
        Matrix<T> eh(half); 
        add(a, d, ad.view()); // ad = a+d
        add(e, h, eh.view()); // eh = e+h
        strassenMP(ad.view(), eh.view(), m1_view); 
    }

    // M2 = (c+d)e
    #pragma omp task
    {
        Matrix<T> cd(half); 
        add(c, d, cd.view()); // cd = c+d
        strassenMP(cd.view(), e, m2_view); 
    }

    // M3 = a(f-h)
    #pragma omp task
    {
        Matrix<T> fh(half); 
        sub(f, h, fh.view()); // fh = f-h
        strassenMP(a, fh.view(), m3_view); 
    }

    // M4 = d(g-e)
    #pragma omp task
    {
        Matrix<T> ge(half); 
        sub(g, e, ge.view()); // ge = g-e
        strassenMP(d, ge.view(), m4_view);
    }

    // M5 = (a+b)h
    #pragma omp task
    {
        Matrix<T> ab(half); 
        add(a, b, ab.view()); // ab = a+b
        strassenMP(ab.view(), h, m5_view);
    }

    // M6 = (c-a)(e+f)
    #pragma omp task
    {
        Matrix<T> ca(half), ef(half);
        sub(c, a, ca.view()); // ca = c-a
        add(e, f, ef.view()); // ef = e+f
        strassenMP(ca.view(), ef.view(), m6_view);
    }

    // M7 = (b-d)(g+h)
    #pragma omp task
    {
        Matrix<T> bd(half), gh(half);
        sub(b, d, bd.view()); // bd = b-d
        add(g, h, gh.view()); // gh = g+h
        strassenMP(bd.view(), gh.view(), m7_view);
    }

    #pragma omp taskwait
    // C12 = M3 + M5
    #pragma omp task
    add(m3_view, m5_view, c12);

    // C21 = M2 + M4
    #pragma omp task
    add(m2_view, m4_view, c21); 
    
    // C11 = M1 + M4 - M5 + M7
    #pragma omp task
    {
        add(m1_view, m4_view, c11);    // c11 = m1 + m4
        sub(c11, m5_view, c11); // c11 = c11 - m5
        add(c11, m7_view, c11); // c11 = c11 + m7
    }

    // C22 = M1 + M3 - M2 + M6
    #pragma omp task
    {
        add(m1_view, m3_view, c22);  // c22 = m1 + m3
        sub(c22, m2_view, c22); // c22 = c22 - m2
        add(c22, m6_view, c22); // c22 = c22 + m6
    }
    #pragma omp taskwait

}

// Hàm hỗ trợ: Tìm lũy thừa của 2 lớn hơn hoặc bằng n
int getNextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Hàm Wrapper: Xử lý Padding trước khi gọi Strassen
template <typename T>
void strassenWithPadding(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C) {
    int n = A.n;
    int m = getNextPowerOfTwo(n);

    // Nếu n đã là lũy thừa của 2, chạy trực tiếp không cần padding
    if (n == m) {
        strassenMP(A, B, C);
        return;
    }

    // 1. Cấp phát ma trận đệm (Padded Matrices) kích thước m x m
    Matrix<T> APad(m);
    Matrix<T> BPad(m);
    Matrix<T> CPad(m);
    
    MatrixView<T> viewAPad = APad.view();
    MatrixView<T> viewBPad = BPad.view();
    MatrixView<T> viewCPad = CPad.view();

    // 2. Copy dữ liệu từ A, B sang APad, BPad và điền số 0 vào phần thừa
    // Sử dụng OpenMP để tăng tốc độ copy
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            if (i < n && j < n) {
                viewAPad.at(i, j) = A.at(i, j);
                viewBPad.at(i, j) = B.at(i, j);
            } else {
                viewAPad.at(i, j) = 0;
                viewBPad.at(i, j) = 0;
            }
            viewCPad.at(i, j) = 0; // Khởi tạo CPad
        }
    }

    // 3. Gọi thuật toán Strassen gốc trên ma trận đã padding (kích thước m x m)
    #pragma omp parallel
    {
        #pragma omp single
        {
            strassenMP(viewAPad, viewBPad, viewCPad);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C.at(i, j) = viewCPad.at(i, j);
        }
    }
}


// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Error: Please Input N" << std::endl;
//         return 1; 
//     }
//     const int N = atoi(argv[1]);
//     if (N <= 0) {
//         std::cerr << "Error: N must be positive" << std::endl;
//         return 1;
//     }

//     srand(time(NULL));
//     std::cout << "Creating matrices A, B, C" << std::endl;
//     Matrix<double>* A = new Matrix<double>(N);
//     Matrix<double>* B = new Matrix<double>(N);
//     Matrix<double>* C = new Matrix<double>(N);
//     Matrix<double>* C_Strassen_Free = new Matrix<double>(N);
//     Matrix<double>* C_Strassen_Class = new Matrix<double>(N);
//     A->fillMatrix(); 
//     B->fillMatrix();
//     MatrixView<double> viewA = A->view();
//     MatrixView<double> viewB = B->view();
//     MatrixView<double> viewC = C->view();
//     MatrixView<double> viewC_Free = C_Strassen_Free->view();
    
//     // -----Naive-------
//     std::cout << "\n--- Running Naive ---" << std::endl;
//     double start_time = omp_get_wtime();
//     naiveMultiplyMP(viewA, viewB, viewC);
//     double end_time = omp_get_wtime();
//     double time_taken = end_time - start_time;
//     std::cout << "Time Taken (Naive): " << time_taken << " seconds" << std::endl;

//     if (N <= 10) {
//         std::cout << "--- Matrix A ---" << std::endl;
//         A->printMatrix();
//         std::cout << "--- Matrix B ---" << std::endl;
//         B->printMatrix();
//         std::cout << "--- Result Matrix C ---" << std::endl;
//         C->printMatrix();
//     }

//     // -----StrassenMP------
//     std::cout << "\n--- Running strassenMP (My Function) ---" << std::endl;
//     double start_time_free = omp_get_wtime();
//     #pragma omp parallel
//     {
//         #pragma omp single
//         {
//             strassenMP(viewA, viewB, viewC_Free);
//         }
//     }
//     double end_time_free = omp_get_wtime();
//     double time_free = end_time_free - start_time_free;
//     std::cout << "Time Taken (strassenMP): " << time_free << " seconds" << std::endl;

//     // ----- Strassen Class-----
//     std::cout << "\n--- Running Matrix::operator* (Class Strassen) ---" << std::endl;
//     double start_time_class = omp_get_wtime();
//     // Gọi toán tử * (dựa trên code Matrix.tpp [cite: 72])
//     *C_Strassen_Class = (*A) * (*B); 
//     double end_time_class = omp_get_wtime();
//     double time_class = end_time_class - start_time_class;
//     std::cout << "Time Taken (Class Strassen): " << time_class << " seconds" << std::endl;


//     std::cout << "\n--- Verification ---" << std::endl;
//     bool test1 = compareMatrices(*C, *C_Strassen_Free);
//     bool test2 = compareMatrices(*C, *C_Strassen_Class);
//     bool test3 = compareMatrices(*C_Strassen_Free, *C_Strassen_Class);

//     std::cout << "Naive vs. strassenMP (Free): " << (test1 ? "SUCCESS" : "FAILURE") << std::endl;
//     std::cout << "Naive vs. Class Strassen:    " << (test2 ? "SUCCESS" : "FAILURE") << std::endl;
//     std::cout << "My Strassen vs. Class Strassen:    " << (test3 ? "SUCCESS" : "FAILURE") << std::endl;

//     if (!test1 || !test2 || !test3) {
//         std::cout << "Error: Results do not match!" << std::endl;
//     } else {
//         std::cout << "All results match." << std::endl;
//     }

//     saveMatrixToFile(*A, "A.txt");
//     saveMatrixToFile(*B, "B.txt");
//     saveMatrixToFile(*C, "C.txt");
//     saveMatrixToFile(*C_Strassen_Free, "C_Strassen_Free.txt");
//     saveMatrixToFile(*C_Strassen_Class, "C_Strassen_Class.txt");

//     delete A;
//     delete B;
//     delete C;
//     return 0;
// }