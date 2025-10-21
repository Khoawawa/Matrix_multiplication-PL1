#include "Matrix.h"
#include <fstream>

int main(){
    std::cout << "\nStarting Performance Measurement..." << std::endl;
    int size = 256;
    
    Matrix<int> A(size);
    Matrix<int> B(size);
    Matrix<int> C_strassen(size);
    Matrix<int> C_cannon(size);
    Matrix<int> C_naive(size);

    A.fillMatrix();
    B.fillMatrix();
    C_strassen.fillMatrix(0);
    C_cannon.fillMatrix(0);
    C_naive.fillMatrix(0);

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

    MatrixView<int> A_view = A.view();
    MatrixView<int> B_view = B.view();
    MatrixView<int> C_strassen_view = C_strassen.view();
    MatrixView<int> C_naive_view = C_naive.view();
    MatrixView<int> C_cannon_view = C_cannon.view();
    
    // Naive
    time_t start_naive = clock();
    Matrix<int>::naiveMultiply(A_view, B_view, C_naive_view);
    time_t end_naive = clock();
    std::cout << "Naive time: " << (double)(end_naive - start_naive) / CLOCKS_PER_SEC << std::endl;
    
    // Cannon
    time_t start_cannon = clock();
    Matrix<int>::cannonMultiply(A_view, B_view, C_cannon_view);
    time_t end_cannon = clock();
    std::cout << "Cannon time: " << (double)(end_cannon - start_cannon) / CLOCKS_PER_SEC << std::endl;

    // Strassen
    time_t start_strassen = clock();
    C_strassen = A * B;
    time_t end_strassen = clock();
    std::cout << "Strassen time: " << (double)(end_strassen - start_strassen) / CLOCKS_PER_SEC << std::endl;

    
    std::cout << "\nPerformance measurement finished." << std::endl;
    std::cout << "Writing result matrices to output files..." << std::endl;

    //Write matrix C to input.txt
    std::ofstream outputFile1("output/outputStrassen.txt");
    if (outputFile1.is_open()) {
        C_strassen.writeData(outputFile1);
        outputFile1.close();
        std::cout << "Successfully wrote result matrix to outputStrassen.txt." << std::endl;
    } else {
        std::cerr << "Error: Unable to open outputStrassen.txt for writing." << std::endl;
    }

    std::ofstream outputFile2("output/outputNaive.txt");
    if (outputFile2.is_open()) {
        C_naive.writeData(outputFile2);
        outputFile2.close();
        std::cout << "Successfully wrote result matrix to outputNaive.txt." << std::endl;
    } else {
        std::cerr << "Error: Unable to open outputNaive.txt for writing." << std::endl;
    }
    
    std::ofstream outputFile3("output/outputCannon.txt");
    if (outputFile3.is_open()) {
        C_cannon.writeData(outputFile3);
        outputFile3.close();
        std::cout << "Successfully wrote result matrix to outputCannon.txt." << std::endl;
    } else {
        std::cerr << "Error: Unable to open outputCannon.txt for writing." << std::endl;
    }
    
    std::cout << "All tasks complete." << std::endl;
    
    return 0;
}