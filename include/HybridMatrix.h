#ifndef HYBRID_MATRIX_H
#define HYBRID_MATRIX_H
#include "Matrix.h"
#include <mpi.h>
#include <queue>
enum Tags{
    TAG_REQUEST_TASK = 1,
    TAG_SEND_ID = 2,
    TAG_SEND_N = 3,
    TAG_SEND_A_DATA = 4,
    TAG_SEND_B_DATA = 5,
    TAG_NO_MORE_TASKS = 6,
    TAG_RESULT_ID = 7,
    TAG_RESULT_DATA = 8
};
template<typename T>
struct StrassenTask{
    int id;
    Matrix<T> A_part;
    Matrix<T> B_part;
    StrassenTask(int id, int n){
        this->id = id;
        A_part = Matrix<T>(n);
        B_part = new Matrix<T>(n);
    }
    StrassenTask(): A_part(), B_part(){};
    StrassenTask(int id, Matrix<T> A_part, Matrix<T> B_part) : id(id), A_part(A_part), B_part(B_part){};
};
//wrapper class
template<typename T>
class HybridMatrix{
    Matrix<T> matrix;
public:
    HybridMatrix(int n) : matrix(Matrix<T>(n)){};
    ~HybridMatrix(){};
    void strassenMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C);
    Matrix<T> operator*(Matrix<T>& B);
    HybridMatrix<T>& operator=(const Matrix<T>& B);
    static Matrix<T> ompStrassenMult(const MatrixView<T>& A, const MatrixView<T>& B);
    MatrixView<T>* splitMatrix4View(const Matrix<T>& M);
    static Matrix<T>* splitMatrix4(const Matrix<T>& M);
    static Matrix<T> add_matrix(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> sub_matrix(const Matrix<T>& A, const Matrix<T>& B);
    static void sendSubMatrices(MatrixView<T>* A, MatrixView<T>* B, MatrixView<T>* C);
    static void sendTask(const StrassenTask<T>& task, int dest);
    static StrassenTask<T> recvTask(int src, int rank);
    void printMatrix(){this->matrix.printMatrix();};
    Matrix<T>& getMatrix(){return this->matrix;}
};
template<typename T>
MPI_Datatype get_mpi_data_type();
template<typename T>
void worker(int rank);
template <typename T>
void coordinator(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int size, Matrix<T>* Ms);
#include "HybridMatrix.tpp"
#endif // HYBRID_MATRIX_H