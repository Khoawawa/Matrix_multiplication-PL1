#include "HybridMatrix.h"
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Process " << rank << " started." << std::endl;
    HybridMatrix<int> A(4), B(4), C(4);
    if (rank == 0){
        A.getMatrix().fillMatrix(1);
        B.getMatrix().fillMatrix(2);
    }
    C = A * B.getMatrix();
    
    if (rank == 0) {
        C.printMatrix();
    }
    MPI_Finalize();
    return 0;
}