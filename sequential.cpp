#include <bits/stdc++.h>
#include "Matrix.h"

// Naive sequential matrix multiplication
template<typename T>
Matrix<T> naive(const Matrix<T>& A, const Matrix<T>& B) {
    int n = A.get_n();
    if (n != B.get_n()) {
        std::cerr << "Error: matrix dimensions must match for multiplication!\n";
        return Matrix<T>(0);
    }

    Matrix<T> res(n);
    T sum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A.get(i, k) * B.get(k, j);
            }
            res.get(i, j) = sum;
        }
    }
    return res;
}

template<typename T>
Matrix<T> strassen(const Matrix<T>& A, const Matrix<T>& B) {
    int n = A.get_n();
    if (n != B.get_n()) {
        std::cerr << "Error: matrix dimensions must match for multiplication!\n";
        return Matrix<T>(0);
    }
    if (n == 1) {
        Matrix<T> res(1);
        res.get(0, 0) = A.get(0, 0) * B.get(0, 0);
        return res;
    }
    int k = n/2;
    Matrix<T> A11(k), A12(k), A21(k), A22(k);
    Matrix<T> B11(k), B12(k), B21(k), B22(k);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            A11.get(i, j) = A.get(i, j);
            A12.get(i, j) = A.get(i, j + k);
            A21.get(i, j) = A.get(i + k, j);
            A22.get(i, j) = A.get(i + k, j + k);

            B11.get(i, j) = B.get(i, j);
            B12.get(i, j) = B.get(i, j + k);
            B21.get(i, j) = B.get(i + k, j);
            B22.get(i, j) = B.get(i + k, j + k);
        }
    }
    Matrix<T> M1 = strassen(A11 + A22, B11 + B22);
    Matrix<T> M2 = strassen(A21 + A22, B11);
    Matrix<T> M3 = strassen(A11, B12 - B22);
    Matrix<T> M4 = strassen(A22, B21 - B11);
    Matrix<T> M5 = strassen(A11 + A12, B22);
    Matrix<T> M6 = strassen(A21 - A11, B11 + B12);
    Matrix<T> M7 = strassen(A12 - A22, B21 + B22);

    Matrix<T> C11 = M1 + M4 - M5 + M7;
    Matrix<T> C12 = M3 + M5;
    Matrix<T> C21 = M2 + M4;
    Matrix<T> C22 = M1 - M2 + M3 + M6;

    Matrix<T> res(n);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            res.get(i, j) = C11.get(i, j);
            res.get(i, j + k) = C12.get(i, j);
            res.get(i + k, j) = C21.get(i, j);
            res.get(i + k, j + k) = C22.get(i, j);
        }
    }
    return res;
}

int main() {
    srand(time(nullptr));

    int n;
    std::cout << "Enter matrix size (n x n): ";
    std::cin >> n;

    Matrix<float> A(n), B(n);

    // Fill matrices with random numbers
    A.fillMatrix();
    B.fillMatrix();

    std::cout << "\nMatrix A:\n";
    A.printMatrix();
    std::cout << "\nMatrix B:\n";
    B.printMatrix();

    // Multiply using naive algorithm
    clock_t start, end;
    start = clock();
    auto C = naive(A, B);
    end = clock();
    double time = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "\nSequential Naive Runtime: " << time << " seconds\n";

    start = clock();
    auto D = strassen(A, B);
    end = clock();
    time = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "\nSequential Naive Runtime: " << time << " seconds\n";

    std::cout << "\nResult (A * B):\n";
    C.printMatrix();
    D.printMatrix();

    return 0;
}
