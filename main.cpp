#include "Matrix.h"

int main(){
    int size = 256;
    Matrix<int> A(size);
    Matrix<int> B(size);
    Matrix<int> C(size);
    
    A.fillMatrix();
    B.fillMatrix();
    C.fillMatrix(0);
    Matrix<int>C_copy = Matrix<int>(C);
    MatrixView<int> A_view = A.view(), B_view = B.view(), C_view = C.view(), C_copy_view = C_copy.view();
    // std::cout << "Matrix A:" << std::endl;
    // A.printMatrix();
    // std::cout << "Matrix B:" << std::endl;
    // B.printMatrix();
    time_t start_naive = clock();
    Matrix<int>::naiveMultiply(A_view, B_view, C_copy_view);
    time_t end_naive = clock();
    std::cout << "Naive time: " << (double)(end_naive - start_naive) / CLOCKS_PER_SEC << std::endl;
    time_t start_strassen = clock();
    C = A * B;
    time_t end_strassen = clock();
    std::cout << "Strassen time: " << (double)(end_strassen - start_strassen) / CLOCKS_PER_SEC << std::endl;
    if (C == C_copy){
        std::cout << "Multiplication result is correct." << std::endl;
    } else {
        std::cout << "Multiplication result is incorrect!" << std::endl;
    }
    // std::cout << "Matrix C:" << std::endl;
    // C.printMatrix();
    return 0;
}