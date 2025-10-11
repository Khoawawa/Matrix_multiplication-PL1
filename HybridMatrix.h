#ifndef HYBRID_MATRIX_H
#define HYBRID_MATRIX_H
#include "Matrix.h"
#include <mpi.h>
template<typename T>
class HybridMatrix : public Matrix<T>{
public:
    using Matrix<T>::Matrix;
    void strassenMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C);
    Matrix<T> operator*(Matrix<T>& B);
    MatrixView<T>* splitMatrix4View();
    Matrix<T>* splitMatrix4();
    static void sendSubMatrices(MatrixView<T>* A, MatrixView<T>* B, MatrixView<T>* C);
};

#include "HybridMatrix.tpp"
#endif // HYBRID_MATRIX_H