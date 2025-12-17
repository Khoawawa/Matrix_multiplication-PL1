#ifndef MPI_STRASSEN_H
#define MPI_STRASSEN_H

#include "Matrix.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

enum MPITags {
    TAG_A_PART = 10,
    TAG_B_PART = 11,
    TAG_RESULT = 12,
    TAG_WORK = 13
};

template<typename T>
MPI_Datatype get_mpi_type();

template<typename T>
class MPIStrassen {
private:
    Matrix<T> matrix;

    static inline int idx(int r, int c, int n) { return r * n + c; }

    static void matrix_add(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n);
    static void matrix_sub(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n);
    static void standard_multiply_serial(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n);

    static void pad_matrix_local(const std::vector<T>& src, std::vector<T>& dest, int n, int m);
    static void unpad_matrix_local(const std::vector<T>& src, std::vector<T>& dest, int n, int m);

    static void strassen_recursive_local(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n);

public:
    MPIStrassen(int n) : matrix(Matrix<T>(n)) {};
    MPIStrassen(const Matrix<T>& M) : matrix(M) {};
    ~MPIStrassen() {};
    Matrix<T>& getMatrix() { return this->matrix; }
    void printMatrix() const { this->matrix.printMatrix(); }


    Matrix<T> operator*(MPIStrassen<T>& B);
    
    static void workerLoop();
};

#include "MPIStrassen.tpp"

#endif