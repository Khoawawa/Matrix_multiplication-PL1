#include "TestCorrectness.h"
#include <ctime>
#include <cstdlib>

void printTestResult(const std::string& testName, bool success) {
    std::cout << "  [Test] " << testName << ": " 
              << (success ? "PASS" : "FAIL") << std::endl;
}

bool testAlgorithmComparison(int size) {
    std::cout << "  Running Comparison Test (Naive vs Strassen vs Cannon)..." << std::endl;
    Matrix<int> A(size);
    Matrix<int> B(size);
    Matrix<int> C_naive(size);
    Matrix<int> C_strassen(size);
    Matrix<int> C_cannon(size);

    A.fillMatrix();
    B.fillMatrix();

    MatrixView<int> A_view = A.view();
    MatrixView<int> B_view = B.view();
    MatrixView<int> C_naive_view = C_naive.view();
    MatrixView<int> C_strassen_view = C_strassen.view();
    MatrixView<int> C_cannon_view = C_cannon.view();

    //Naive
    Matrix<int>::naiveMultiply(A.view(), B.view(), C_naive_view);

    //Strassen
    C_strassen = A * B;

    //Cannon
    Matrix<int>::cannonMultiply(A.view(), B.view(), C_cannon_view);

    // Compare
    bool strassen_pass = Matrix<int>::areMatricesEqual(C_naive.view(), C_strassen.view());
    bool cannon_pass = Matrix<int>::areMatricesEqual(C_naive.view(), C_cannon.view());

    printTestResult("Naive vs Strassen", strassen_pass);
    printTestResult("Naive vs Cannon", cannon_pass);

    return strassen_pass && cannon_pass;
}

bool testIdentityMultiplication(int size) {
    std::cout << "  Running Identity Multiplication Test (A * I = A)..." << std::endl;
    Matrix<int> A(size);
    Matrix<int> I(size);
    Matrix<int> C_strassen(size);
    Matrix<int> C_naive(size);
    Matrix<int> C_cannon(size);

    A.fillMatrix();
    I.createIdentityMatrix();

    MatrixView<int> A_view = A.view();
    MatrixView<int> C_naive_view = C_naive.view();
    MatrixView<int> C_strassen_view = C_strassen.view();
    MatrixView<int> C_cannon_view = C_cannon.view();

    // Strassen
    C_strassen = A * I;
    bool strassen_pass = Matrix<int>::areMatricesEqual(A.view(), C_strassen_view);
    printTestResult("Strassen (A * I = A)", strassen_pass);

    // Naive
    Matrix<int>::naiveMultiply(A.view(), I.view(), C_naive_view);
    bool naive_pass = Matrix<int>::areMatricesEqual(A.view(), C_naive_view);
    printTestResult("Naive (A * I = A)", naive_pass);
    
    return strassen_pass && naive_pass;
}

bool testZeroMultiplication(int size) {
    std::cout << "  Running Zero Multiplication Test (A * 0 = 0)..." << std::endl;
    Matrix<int> A(size);
    Matrix<int> Z(size);
    Matrix<int> C_expected_zero(size);
    Matrix<int> C_result_strassen(size);
    Matrix<int> C_result_naive(size);

    A.fillMatrix();
    Z.fillMatrix(0);
    C_expected_zero.fillMatrix(0);

    MatrixView<int> C_naive_view = C_result_naive.view();

    // Strassen
    C_result_strassen = A * Z;
    bool strassen_pass = Matrix<int>::areMatricesEqual(C_expected_zero.view(), C_result_strassen.view());
    printTestResult("Strassen (A * 0 = 0)", strassen_pass);

    // Naive
    Matrix<int>::naiveMultiply(A.view(), Z.view(), C_naive_view);
    bool naive_pass = Matrix<int>::areMatricesEqual(C_expected_zero.view(), C_result_naive.view());
    printTestResult("Naive (A * 0 = 0)", naive_pass);

    return strassen_pass && naive_pass;
}

void runFullTestSuite() {
    std::cout << "=========================================" << std::endl;
    std::cout << "     RUNNING MATRIX CORRECTNESS SUITE    " << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    int totalTests = 0;
    int passedTests = 0;
    bool all_pass = true;

    // Array about sizes that need for testing
    int sizesToTest[] = {
        1,
        9,
        64,
        66,
        128,
        130
    };
    
    for (int size : sizesToTest) {
        std::cout << "\n--- Testing for Size: " << size << "x" << size << " ---" << std::endl;
        totalTests = 0;
        passedTests = 0;

        bool r1 = testAlgorithmComparison(size);
        totalTests++; if (r1) passedTests++;

        bool r2 = testIdentityMultiplication(size);
        totalTests++; if (r2) passedTests++;
        
        bool r3 = testZeroMultiplication(size);
        totalTests++; if (r3) passedTests++;

        std::cout << "--- Summary for Size " << size << ": " 
                  << passedTests << " / " << totalTests << " tests passed ---" << std::endl;
        if (passedTests != totalTests) {
            all_pass = false;
        }
    }

    std::cout << "=========================================" << std::endl;
    if (all_pass) {
        std::cout << "FINAL RESULT: ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "FINAL RESULT: SOME TESTS FAILED!" << std::endl;
    }
    std::cout << "=========================================" << std::endl;
}

/**
 * @brief Main function.
 */
int main() {
    // random
    srand(static_cast<unsigned int>(time(NULL))); 
    
    runFullTestSuite();
    
    return 0;
}