#include "Matrix.h"
#include <fstream>

int main(){
    std::cout << "\nStarting Performance Measurement..." << std::endl;
    int size = 300;
    Matrix<int> A(size);
    Matrix<int> B(size);
    Matrix<int> C(size);
    
    A.fillMatrix();
    B.fillMatrix();
    C.fillMatrix(0);

    //Write matrix A and matrix B to input.txt
    std::cout << "Writing matrices A and B to input.txt..." << std::endl;
    std::ofstream inputFile("input/input.txt");
    if (inputFile.is_open()) {
        A.writeData(inputFile);
        inputFile << std::endl;
        B.writeData(inputFile);
        inputFile.close();
        std::cout << "Successfully wrote to input.txt." << std::endl;
    } else {
        std::cerr << "Error: Unable to open input.txt for writing." << std::endl;
    }

    Matrix<int>C_copy = Matrix<int>(C);
    MatrixView<int> A_view = A.view(), B_view = B.view(), C_view = C.view(), C_copy_view = C_copy.view();
    
    time_t start_naive = clock();
    Matrix<int>::naiveMultiply(A_view, B_view, C_copy_view);
    time_t end_naive = clock();
    std::cout << "Naive time: " << (double)(end_naive - start_naive) / CLOCKS_PER_SEC << std::endl;
    
    time_t start_strassen = clock();
    C = A * B;
    time_t end_strassen = clock();
    std::cout << "Strassen time: " << (double)(end_strassen - start_strassen) / CLOCKS_PER_SEC << std::endl;



    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Running test correctness for size = " << size << "x" << size << std::endl;
    if (Matrix<int>::areMatricesEqual(C_copy_view, C)) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;

    //"Write matrix C to input.txt
    std::cout << "Writing result matrices C and C_copy to output.txt..." << std::endl;
    std::ofstream outputFile1("output/outputStrassen.txt");
    if (outputFile1.is_open()) {
        C.writeData(outputFile1);
        outputFile1.close();
        std::cout << "Successfully wrote result matrix to outputStrassen.txt." << std::endl;
    } else {
        std::cerr << "Error: Unable to open outputStrassen.txt for writing." << std::endl;
    }

    std::ofstream outputFile2("output/outputNaive.txt");
    if (outputFile2.is_open()) {
        C_copy.writeData(outputFile2);
        outputFile2.close();
        std::cout << "Successfully wrote result matrix to outputNaive.txt." << std::endl;
    } else {
        std::cerr << "Error: Unable to open outputNaive.txt for writing." << std::endl;
    }
    
    
    return 0;
}