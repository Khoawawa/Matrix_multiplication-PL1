#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <omp.h>

// Include file chứa thuật toán Strassen và Naive của bạn
// Đảm bảo bạn đã xóa hàm main() trong file omp.h này
#include "omp.h" 
#include "Matrix.h"

using namespace std;

// ==========================================
// CÁC HÀM HỖ TRỢ TEST
// ==========================================

void printTestResult(const std::string& testName, bool success, double timeSec = -1.0) {
    std::cout << "  [Test] " << std::left << std::setw(45) << testName << ": ";
    if (success) {
        std::cout << "\033[1;32mPASS\033[0m"; // Màu xanh lá
    } else {
        std::cout << "\033[1;31mFAIL\033[0m"; // Màu đỏ
    }
    
    if (timeSec >= 0) {
        std::cout << " (" << std::fixed << std::setprecision(4) << timeSec << "s)";
    }
    std::cout << std::endl;
}

// Hàm so sánh ma trận (chấp nhận sai số nhỏ do kiểu double)
bool checkMatricesEqual(const Matrix<double>& A, const Matrix<double>& B, double epsilon = 1e-5) {
    if (A.get_n() != B.get_n()) return false;
    int n = A.get_n();
    bool equal = true;
    
    #pragma omp parallel for collapse(2) shared(equal)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!equal) continue; // Skip nếu đã tìm thấy lỗi
            if (std::abs(A.get(i, j) - B.get(i, j)) > epsilon) {
                #pragma omp critical
                {
                    // Chỉ in lỗi đầu tiên tìm thấy
                    if (equal) {
                        // std::cerr << "Diff at (" << i << "," << j << "): " << A.get(i,j) << " vs " << B.get(i,j) << std::endl;
                        equal = false;
                    }
                }
            }
        }
    }
    return equal;
}

// ==========================================
// CÁC KỊCH BẢN TEST (TEST CASES)
// ==========================================

bool testAlgorithmComparison(int size) {
    std::cout << "  Running Comparison (Naive vs Strassen Padding)..." << std::endl;
    
    Matrix<double> A(size);
    Matrix<double> B(size);
    Matrix<double> C_naive(size);
    Matrix<double> C_strassen(size);

    A.fillMatrix(); // Random numbers
    B.fillMatrix();

    MatrixView<double> vA = A.view();
    MatrixView<double> vB = B.view();
    MatrixView<double> vCn = C_naive.view();
    MatrixView<double> vCs = C_strassen.view();

    // 1. Chạy Naive (Sử dụng bản Parallel OMP có sẵn trong omp.h để làm chuẩn nhanh hơn)
    // Lưu ý: Với size > 3000, ngay cả Naive Parallel cũng mất thời gian.
    double t1 = omp_get_wtime();
    naiveMultiplyMP(vA, vB, vCn); 
    double time_naive = omp_get_wtime() - t1;

    // 2. Chạy Strassen (Bản có Padding)
    double t2 = omp_get_wtime();
    strassenWithPadding(vA, vB, vCs);
    double time_strassen = omp_get_wtime() - t2;

    // 3. So sánh
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
    
    // Tạo ma trận đơn vị thủ công hoặc dùng hàm nếu có
    // Giả sử làm thủ công cho chắc chắn
    MatrixView<double> vI = I.view();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<size; i++) 
        for(int j=0; j<size; j++) 
            vI.at(i,j) = (i==j) ? 1.0 : 0.0;

    MatrixView<double> vA = A.view();
    MatrixView<double> vC = C_result.view();

    // Chạy Strassen
    strassenWithPadding(vA, vI, vC);

    bool pass = checkMatricesEqual(A, C_result);
    printTestResult("Identity (A*I=A)", pass);
    return pass;
}

bool testZeroMultiplication(int size) {
    std::cout << "  Running Zero Test (A * 0 = 0)..." << std::endl;
    Matrix<double> A(size);
    Matrix<double> Z(size); // Mặc định là 0
    Matrix<double> C_result(size);
    
    // Đảm bảo Z toàn số 0
    MatrixView<double> vZ = Z.view();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<size; i++) 
        for(int j=0; j<size; j++) 
            vZ.at(i,j) = 0.0;

    A.fillMatrix();
    
    MatrixView<double> vA = A.view();
    MatrixView<double> vC = C_result.view();

    // Chạy Strassen
    strassenWithPadding(vA, vZ, vC);

    // Kiểm tra kết quả có toàn 0 không
    bool pass = checkMatricesEqual(Z, C_result); // So sánh với Z
    printTestResult("Zero (A*0=0)", pass);
    return pass;
}

// ==========================================
// MAIN TEST SUITE
// ==========================================

void runFullTestSuite() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   OPENMP STRASSEN CORRECTNESS SUITE     " << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Kiểm tra số luồng đang chạy
    int n_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        n_threads = omp_get_num_threads();
    }
    std::cout << "Running with " << n_threads << " OpenMP threads.\n" << std::endl;

    bool all_pass = true;

    // Danh sách các size cần test
    // Bao gồm: size nhỏ, size lẻ (test padding), lũy thừa 2, và size lớn
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
        4000,
        4096   // Size lớn ~3000 (Padding lên 4096) -- Case nặng nhất
    };

    for (int size : sizesToTest) {
        std::cout << "\n-----------------------------------------" << std::endl;
        std::cout << ">>> Testing Size: " << size << "x" << size << std::endl;
        std::cout << "-----------------------------------------" << std::endl;

        int totalTests = 0;
        int passedTests = 0;

        // 1. Identity Test (Nhanh)
        bool r1 = testIdentityMultiplication(size);
        totalTests++; if (r1) passedTests++;

        // 2. Zero Test (Nhanh)
        bool r2 = testZeroMultiplication(size);
        totalTests++; if (r2) passedTests++;

        // 3. Comparison Test (Chậm hơn do phải chạy cả Naive để so sánh)
        // Với size 4000, Naive O(N^3) chạy khá lâu kể cả song song.
        // Có thể skip verify Naive cho size quá lớn nếu muốn tiết kiệm thời gian,
        // nhưng ở đây mình vẫn để chạy để đảm bảo chính xác tuyệt đối.
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
    // Thiết lập random seed
    srand(static_cast<unsigned int>(time(NULL))); 
    
    // Tùy chọn: Set số luồng cố định nếu cần
    // omp_set_num_threads(4);

    runFullTestSuite();
    
    return 0;
}