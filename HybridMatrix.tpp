#include "HybridMatrix.h"
template<typename T>
MatrixView<T>* HybridMatrix<T>::splitMatrix4View()
{
    int half = this->n / 2;

    MatrixView<T> a11{this->data, half, this->n};
    MatrixView<T> a12{this->data + half, half, this->n};
    MatrixView<T> a21{this->data + half * this->n, half, this->n};
    MatrixView<T> a22{this->data + half * this->n + half, half, this->n};

    MatrixView<T>* views = new MatrixView<T>[4];
    views[0] = a11;
    views[1] = a12;
    views[2] = a21;
    views[3] = a22;
    return views;
}
template<typename T> 
Matrix<T> * HybridMatrix<T>::splitMatrix4()
{
    Matrix<T> * matrices = new Matrix<T>[4];
    matrices[0] = Matrix<T>()
}
template<typename T>
void HybridMatrix<T>::strassenMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C)
{

}
template<typename T>
Matrix<T> HybridMatrix<T>::operator*(Matrix<T>& B){
    Matrix<T> C(this->n);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = this->get_n();
    
    if (n <= 64 || size==1){
        Matrix<T>::naiveMultiply(*this, B, C_view);
    }

    if(rank == 0){
        MatrixView<T>* A_views = this->splitMatrix4View();
        MatrixView<T>* B_views = B.splitMatrix4View();
        MatrixView<T>* C_views = C.splitMatrix4View();

        HybridMatrix::sendSubMatrices(A_views, B_views, C_views);
    }

}
