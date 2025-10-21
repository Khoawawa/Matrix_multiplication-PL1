#include "Matrix.h"

template <typename T>
void Matrix<T>::fillMatrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            data[i * n + j] = rand() % 5;
        }
    }
}
template <typename T>
void Matrix<T>::fillMatrix(T val)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            data[i * n + j] = val;
        }
    }
}
template <typename T>
void Matrix<T>::printMatrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << data[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}
template <typename T>
Matrix<T>::operator MatrixView<T>()
{
    return MatrixView<T>{this->data, n, n};
}
template <typename T>
Matrix<T>::operator MatrixView<T>() const
{
    return MatrixView<T>{this->data, n, n};
}
template <typename T>
void Matrix<T>::add(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> C)
{
    for (int i = 0; i < A.n; i++)
    {
        for (int j = 0; j < A.n; j++)
        {
            C.at(i, j) = A.at(i, j) + B.at(i, j);
        }
    }
}
template <typename T>
void Matrix<T>::sub(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> C)
{
    for (int i = 0; i < A.n; i++)
    {
        for (int j = 0; j < A.n; j++)
        {
            C.at(i, j) = A.at(i, j) - B.at(i, j);
        }
    }
}
template <typename T> void Matrix<T>::naiveMultiply(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C)
{
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

template <typename T>
void Matrix<T>::cannonMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C)
{
    const int n = A.n;
    const int BLOCK_SIZE = 32; // A sensible block size to define task granularity

    #pragma omp parallel
    {
        // Have only a single thread generate all the tasks
        #pragma omp single
        {
            // Iterate over the matrix C in blocks
            for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
                for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
                    
                    // Create one task for each block of C
                    #pragma omp task
                    {
                        // This is the code that each task will execute
                        // It computes one block of the final matrix C
                        int i_end = std::min(bi + BLOCK_SIZE, n);
                        for (int i = bi; i < i_end; ++i) {
                            int j_end = std::min(bj + BLOCK_SIZE, n);
                            for (int j = bj; j < j_end; ++j) {
                                T sum = 0;
                                // Innermost loop computes the dot product for C[i, j]
                                for (int k = 0; k < n; ++k) {
                                    int a_col = (j + i + k) % n;
                                    int b_row = (i + j + k) % n;
                                    sum += A.at(i, a_col) * B.at(b_row, j);
                                }
                                C.at(i, j) = sum;
                            }
                        }
                    } // End of task
                }
            }
        } // All tasks have been created by the single thread
    } // The parallel region implicitly waits for all tasks to complete before ending
}



template <typename T>
MatrixView<T> Matrix<T>::view()
{
    return MatrixView<T>(data, n, n);
}
template <typename T>
void Matrix<T>::strassenMultiply(const MatrixView<T> &A, const MatrixView<T> &B, MatrixView<T> &C)
{
    if (A.n <= 64)
    {
        // TODO: either naive way or the thingy
        // std::cout << "Im here";
        Matrix<T>::naiveMultiply(A, B, C);
        return;
    }
    // first thing first get the 8 slice of the matrix, corresponding to a..h
    // if we were to create a new matrix each time --> memory inefficient on big n
    // rather than that how about a view? -> like torch tensor
    int half = A.n / 2;
    // Split A into 4 submatrices
    MatrixView<T> a{A.data, half, A.stride};
    MatrixView<T> b{A.data + half, half, A.stride};
    MatrixView<T> c{A.data + half * A.stride, half, A.stride};
    MatrixView<T> d{A.data + half * A.stride + half, half, A.stride};

    // Split B into 4 submatrices
    MatrixView<T> e{B.data, half, B.stride};
    MatrixView<T> f{B.data + half, half, B.stride};
    MatrixView<T> g{B.data + half * B.stride, half, B.stride};
    MatrixView<T> h{B.data + half * B.stride + half, half, B.stride};

    // split C into 4 submatrices
    MatrixView<T> c11{C.data, half, C.stride};
    MatrixView<T> c12{C.data + half, half, C.stride};
    MatrixView<T> c21{C.data + half * C.stride, half, C.stride};
    MatrixView<T> c22{C.data + half * C.stride + half, half, C.stride};

    // compute M1​=(a+d)(e+h)
    Matrix<T> m1(half);
    MatrixView<T> m1_view = m1.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> ad(half);
        Matrix<T> eh(half);
        add(a, d, ad);
        add(e, h, eh);
        this->strassenMultiply(ad, eh, m1_view);
        // std::cout << "[Thread " << tid << "] << M1 computed" << std::endl;
    }
    // compute M2​=(c+d)e
    Matrix<T> m2(half);
    MatrixView<T> m2_view = m2.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> cd(half);
        add(c, d, cd);
        this->strassenMultiply(cd, e, m2_view);
        // std::cout << "[Thread " << tid << "] << M2 computed" << std::endl;
    }
    // compute M3​=a(f−h)
    Matrix<T> m3(half);
    MatrixView<T> m3_view = m3.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> fh(half);
        sub(f, h, fh);
        this->strassenMultiply(a, fh, m3_view);
        // std::cout << "[Thread " << tid << "] << M3 computed" << std::endl;
    }
    // compute M4=d(g−e)
    Matrix<T> m4(half);
    MatrixView<T> m4_view = m4.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> ge(half);
        sub(g, e, ge);
        this->strassenMultiply(d, ge, m4_view);
        // std::cout << "[Thread " << tid << "] << M4 computed" << std::endl;
    }
    // compute M5=(a+b)h
    Matrix<T> m5(half);
    MatrixView<T> m5_view = m5.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> ab(half);
        add(a, b, ab);
        this->strassenMultiply(ab, h, m5_view);
        // std::cout << "[Thread " << tid << "] << M5 computed" << std::endl;
    }
    // compute M6=(c−a)(e+f)
    Matrix<T> m6(half);
    MatrixView<T> m6_view = m6.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> ca(half), ef(half);
        sub(c, a, ca);
        add(e, f, ef);
        this->strassenMultiply(ca, ef, m6_view);
        // std::cout << "[Thread " << tid << "] << M6 computed" << std::endl;
    }
    // compute M7=(b−d)(g+h)
    Matrix<T> m7(half);
    MatrixView<T> m7_view = m7.view();
#pragma omp task
    {
        int tid = omp_get_thread_num();
        Matrix<T> bd(half), gh(half);
        sub(b, d, bd);
        add(g, h, gh);
        this->strassenMultiply(bd, gh, m7_view);
        // std::cout << "[Thread " << tid << "] << M7 computed" << std::endl;
    }

#pragma omp taskwait
    // std::cout <<"finised tasks m1..m7\n";

    // m1.printMatrix();
    // m2.printMatrix();
    // m3.printMatrix();
    // m4.printMatrix();
    // m5.printMatrix();
    // m6.printMatrix();
    // m7.printMatrix();

    // C = [
    //      M1 + M4 - M5 + M7 | M3 + M5
    //      M2 +  M4 | M1 + M3 - M2 + M6
    // ]
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 + M3 - M2 + M6
    Matrix<T> m14(half), m145(half);
    add(m1, m4, m14);
    sub(m14, m5, m145);
    add(m145, m7, c11);

    add(m3, m5, c12);
    add(m2, m4, c21);

    Matrix<T> m13(half), m132(half);
    add(m1, m3, m13);
    sub(m13, m2, m132);
    add(m132, m6, c22);
    // #pragma omp task
    // {
    //     Matrix<T> m14(half), m145(half);
    //     add(m1, m4, m14);
    //     sub(m14, m5, m145);
    //     add(m145, m7, c11);
    //     std::cout << "[Thread " << omp_get_thread_num() << "] << c11 computed" << std::endl;
    // }
    // #pragma omp task
    // {
    //     add(m3, m5, c12);
    //     std::cout << "[Thread " << omp_get_thread_num() << "] << c12 computed" << std::endl;
    // }
    // #pragma omp task
    // {
    //     add(m2, m4, c21);
    //     std::cout << "[Thread " << omp_get_thread_num() << "] << c21 computed" << std::endl;
    // }
    // #pragma omp task
    // {
    //     Matrix<T> m13(half), m132(half);
    //     add(m1, m3, m13);
    //     sub(m13, m2, m132);
    //     add(m132, m6, c22);
    //     std::cout << "[Thread " << omp_get_thread_num() << "] << c22 computed" << std::endl;
    // }

    // #pragma omp taskwait
    // std::cout <<"finised tasks c11..c22\n";
}
template <typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> &B)
{
    Matrix<T> C(this->n);
    MatrixView<T> B_view{B.data, B.n, B.n};
    MatrixView<T> C_view{C.data, C.n, C.n};
#pragma omp parallel
    {
#pragma omp single nowait
        {

            this->strassenMultiply(*this, B_view, C_view);
        }
    }
    return C;
}
template <typename T>
int Matrix<T>::get_n() const { return n; }
template <typename T>
T &Matrix<T>::get(int i, int j) const { return data[i * n + j]; }

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &B)
{
    if (this != &B)
    { // Check for self-assignment
        if (n != B.n)
        {
            throw std::invalid_argument("Cannot assign matrices of different sizes");
        }
        std::copy(B.data, B.data + n * n, this->data); // Deep copy
    }
    return *this;
}
template<typename T>
Matrix<T>* Matrix<T>::splitQuadrantMatrix()
{
    int half = this->n / 2;
    Matrix<T>* ret = new Matrix<T>[4]{
        Matrix<T>(half),
        Matrix<T>(half),
        Matrix<T>(half),
        Matrix<T>(half)
    };
    
    for (int pos = 0; pos < 4; pos++)
    {   
        int row_offset = (pos / 2) * half;
        int col_offset = (pos % 2) * half;
        for (int i = 0; i < half; i++)
        {
            std::copy(
                this->data + (row_offset + i) * this-> n + col_offset,
                this->data + (row_offset + i) * this-> n + col_offset + half,
                ret[pos].data + i * half
            );
        }    
        
    }
    return ret;
}
template<typename T>
bool Matrix<T>::areMatricesEqual(const MatrixView<T>& A, const MatrixView<T>& B, double epsilon) {
    if (A.n != B.n) {
        return false;
    }

    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            if (std::abs(A.at(i, j) - B.at(i, j)) > epsilon) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): A=" << A.at(i, j) 
                          << ", B=" << B.at(i, j) << std::endl;
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void Matrix<T>::createIdentityMatrix()
{
    // First, fill the entire matrix with the additive identity (0)
    this->fillMatrix(static_cast<T>(0));
    
    // Then, set the main diagonal to the multiplicative identity (1)
    for (int i = 0; i < n; ++i) {
        data[i * n + i] = static_cast<T>(1);
    }
}

template <typename T>
void Matrix<T>::writeData(std::ostream& os) const {
    os << n << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            os << data[i * n + j] << (j == n - 1 ? "" : " ");
        }
        os << std::endl;
    }
}