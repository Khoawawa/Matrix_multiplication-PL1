#include "include/HybridMatrix.h"
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #pragma omp parallel
{
    printf("Rank %d, Thread %d\n", rank, omp_get_thread_num());
}
    int size = 1000;
    HybridMatrix<int> A(size), B(size), C(size);
    if (rank == 0){
        A.getMatrix().fillMatrix(1);
        B.getMatrix().fillMatrix(2);
    }
    C = A * B.getMatrix();
    
    if (rank == 0) {
        Matrix<int> D = Matrix<int>(size);
        MatrixView<int> D_view = D.view();
        
        Matrix<int>::naiveMultiply(A.getMatrix(), B.getMatrix(), D_view);
        
        if (C.getMatrix() == D){
            std::cout << "Multiplication result is correct." << std::endl;
        } else {
            std::cout << "Multiplication result is incorrect!" << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}