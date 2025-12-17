#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <omp.h>

#include "strassen.h" 
#include "Matrix.h"

using namespace std;

void printTestResult(const std::string& testName, bool success, double timeSec = -1.0) {
    std::cout << "  [Test] " << std::left << std::setw(45) << testName << ": ";
    if (success) {
        std::cout << "\033[1;32mPASS\033[0m";
    } else {
        std::cout << "\033[1;31mFAIL\033[0m"; 
    }
    
    if (timeSec >= 0) {
        std::cout << " (" << std::fixed << std::setprecision(4) << timeSec << "s)";
    }
    std::cout << std::endl;
}

bool checkMatricesEqual(const Matrix<double>& A, const Matrix<double>& B, double epsilon = 1e-5) {
    if (A.get_n() != B.get_n()) return false;
    int n = A.get_n();
    bool equal = true;
    
    #pragma omp parallel for collapse(2) shared(equal)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!equal) continue; 
            if (std::abs(A.get(i, j) - B.get(i, j)) > epsilon) {
                #pragma omp critical
                {
                    if (equal) {
                        equal = false;
                    }
                }
            }
        }
    }
    return equal;
}


bool testAlgorithmComparison(int size) {
    std::cout << "  Running Comparison (Naive vs Strassen Padding)..." << std::endl;
    
    Matrix<double> A(size);
    Matrix<double> B(size);
    Matrix<double> C_naive(size);
    Matrix<double> C_strassen(size);

    A.fillMatrix(); 
    B.fillMatrix();

    MatrixView<double> vA = A.view();
    MatrixView<double> vB = B.view();
    MatrixView<double> vCn = C_naive.view();
    MatrixView<double> vCs = C_strassen.view();

    double t1 = omp_get_wtime();
    naiveMultiplyMP(vA, vB, vCn); 
    double time_naive = omp_get_wtime() - t1;

    double t2 = omp_get_wtime();
    strassenWithPadding(vA, vB, vCs);
    double time_strassen = omp_get_wtime() - t2;

    bool pass = checkMatricesEqual(C_naive, C_strassen);
    
    std::string label = "Naive vs Strassen (Size " + std::to_string(size) + ")";
    printTestResult(label, pass, time_strassen);
    
    if (pass) {
        std::cout << "        -> Speedup: " << std::fixed << std::setprecision(2) 
                  << (time_naive / time_strassen) << "x (Naive: " << time_naive << "s)" << std::endl;
    }

    return pass;
}

bool testIdentityMultiplication(int size) {
    std::cout << "  Running Identity Test (A * I = A)..." << std::endl;
    Matrix<double> A(size);
    Matrix<double> I(size);
    Matrix<double> C_result(size);

    A.fillMatrix();
    
    MatrixView<double> vI = I.view();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<size; i++) 
        for(int j=0; j<size; j++) 
            vI.at(i,j) = (i==j) ? 1.0 : 0.0;

    MatrixView<double> vA = A.view();
    MatrixView<double> vC = C_result.view();

    strassenWithPadding(vA, vI, vC);

    bool pass = checkMatricesEqual(A, C_result);
    printTestResult("Identity (A*I=A)", pass);
    return pass;
}

bool testZeroMultiplication(int size) {
    std::cout << "  Running Zero Test (A * 0 = 0)..." << std::endl;
    Matrix<double> A(size);
    Matrix<double> Z(size);
    Matrix<double> C_result(size);
    
    MatrixView<double> vZ = Z.view();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<size; i++) 
        for(int j=0; j<size; j++) 
            vZ.at(i,j) = 0.0;

    A.fillMatrix();
    
    MatrixView<double> vA = A.view();
    MatrixView<double> vC = C_result.view();

    strassenWithPadding(vA, vZ, vC);

    bool pass = checkMatricesEqual(Z, C_result);
    printTestResult("Zero (A*0=0)", pass);
    return pass;
}

void runFullTestSuite() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   OPENMP STRASSEN CORRECTNESS SUITE     " << std::endl;
    std::cout << "=========================================" << std::endl;
    
    int n_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        n_threads = omp_get_num_threads();
    }
    std::cout << "Running with " << n_threads << " OpenMP threads.\n" << std::endl;

    bool all_pass = true;

    std::vector<int> sizesToTest = {
        // 1,
        // 17,    // Lẻ nhỏ
        // 32,    // Lũy thừa 2
        // 65,    // Lẻ (Padding lên 128)
        // 128,
        // 513,   // Lẻ (Padding lên 1024)
        // 1024,
        // 2048,
        // 2050,  // Size lớn ~2000 (Padding lên 4096)
        // 3000,
        1024,
        4000,
        4096
        // 10000
    };

    for (int size : sizesToTest) {
        std::cout << "\n-----------------------------------------" << std::endl;
        std::cout << ">>> Testing Size: " << size << "x" << size << std::endl;
        std::cout << "-----------------------------------------" << std::endl;

        int totalTests = 0;
        int passedTests = 0;

        bool r1 = testIdentityMultiplication(size);
        totalTests++; if (r1) passedTests++;

        bool r2 = testZeroMultiplication(size);
        totalTests++; if (r2) passedTests++;

        if (size > 10000) {
            std::cout << "  [Skip Comparison] Size too large for Naive verification." << std::endl;
        } else {
            bool r3 = testAlgorithmComparison(size);
            totalTests++; if (r3) passedTests++;
        }

        std::cout << ">>> Summary for Size " << size << ": " 
                  << passedTests << " / " << totalTests << " tests passed." << std::endl;
        
        if (passedTests != totalTests) {
            all_pass = false;
        }
    }

    std::cout << "\n=========================================" << std::endl;
    if (all_pass) {
        std::cout << "\033[1;32mFINAL RESULT: ALL TESTS PASSED!\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31mFINAL RESULT: SOME TESTS FAILED!\033[0m" << std::endl;
    }
    std::cout << "=========================================" << std::endl;
}

int main() {
    srand(static_cast<unsigned int>(time(NULL))); 
    runFullTestSuite();
    return 0;
}