#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "MPIStrassen.h"

using namespace std;

template <typename T>
void standard_multiply_serial(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    int n = A.get_n();
    std::fill(C.get_data(), C.get_data() + n * n, 0);
    
    const T* aData = A.get_data();
    const T* bData = B.get_data();
    T* cData = C.get_data();

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            T temp = aData[i * n + k];
            for (int j = 0; j < n; ++j) {
                cData[i * n + j] += temp * bData[k * n + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024; 
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    MPIStrassen<double> A(N), B(N), C_strassen(N);
    Matrix<double> C_serial(N);

    if (rank == 0) {
        cout << "========================================" << endl;
        cout << "Running MPI Strassen Benchmark" << endl;
        cout << "Matrix Size N = " << N << endl;
        cout << "MPI Processes = " << size << endl;
        cout << "========================================" << endl;

        A.getMatrix().fillMatrix(); 
        B.getMatrix().fillMatrix();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start_strassen = MPI_Wtime();

    if (rank == 0) {
        C_strassen = A * B; 
    } 
    else if (rank <= 7) {
        MPIStrassen<double>::workerLoop();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_strassen = MPI_Wtime();
    double time_strassen = end_strassen - start_strassen;

    if (rank == 0) {
        cout << fixed << setprecision(4);
        cout << "\n[1] Strassen MPI Time: " << time_strassen << " s" << endl;

        double start_serial = MPI_Wtime();
        standard_multiply_serial(A.getMatrix(), B.getMatrix(), C_serial);
        double end_serial = MPI_Wtime();
        double time_serial = end_serial - start_serial;

        cout << "[2] Standard Serial Time: " << time_serial << " s" << endl;

        if (time_strassen > 0) {
            double speedup = time_serial / time_strassen;
            cout << ">> Speedup (Serial / Strassen): " << speedup << "x" << endl;
        }

        cout << "\nChecking correctness..." << endl;
        
        if (C_strassen.getMatrix() == C_serial) {
            cout << ">> SUCCESS: Results match!" << endl;
        } else {
            cout << ">> FAILURE: Results do NOT match!" << endl;
        }
        cout << "========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}