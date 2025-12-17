#include "MPIStrassen.h"

template<typename T>
void MPIStrassen<T>::matrix_add(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = A[i] + B[i];
}

template<typename T>
void MPIStrassen<T>::matrix_sub(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n) {
    for (int i = 0; i < n * n; ++i) C[i] = A[i] - B[i];
}

template<typename T>
void MPIStrassen<T>::standard_multiply_serial(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n) {
    std::fill(C.begin(), C.end(), 0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            T temp = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += temp * B[k * n + j];
            }
        }
    }
}

template<typename T>
void MPIStrassen<T>::strassen_recursive_local(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n) {
    int CROSSOVER_SIZE = 64; 
    if (n <= CROSSOVER_SIZE) {
        standard_multiply_serial(A, B, C, n);
        return;
    }

    int mid = n / 2;
    int q = mid * mid;

    std::vector<T> A11(q), A12(q), A21(q), A22(q);
    std::vector<T> B11(q), B12(q), B21(q), B22(q);
    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            A11[idx(i, j, mid)] = A[idx(i, j, n)];
            A12[idx(i, j, mid)] = A[idx(i, j + mid, n)];
            A21[idx(i, j, mid)] = A[idx(i + mid, j, n)];
            A22[idx(i, j, mid)] = A[idx(i + mid, j + mid, n)];

            B11[idx(i, j, mid)] = B[idx(i, j, n)];
            B12[idx(i, j, mid)] = B[idx(i, j + mid, n)];
            B21[idx(i, j, mid)] = B[idx(i + mid, j, n)];
            B22[idx(i, j, mid)] = B[idx(i + mid, j + mid, n)];
        }
    }

    std::vector<T> P1(q), P2(q), P3(q), P4(q), P5(q), P6(q), P7(q);
    std::vector<T> T1(q), T2(q);

    matrix_sub(B12, B22, T2, mid); strassen_recursive_local(A11, T2, P1, mid);
    matrix_add(A11, A12, T1, mid); strassen_recursive_local(T1, B22, P2, mid);
    matrix_add(A21, A22, T1, mid); strassen_recursive_local(T1, B11, P3, mid);
    matrix_sub(B21, B11, T2, mid); strassen_recursive_local(A22, T2, P4, mid);
    matrix_add(A11, A22, T1, mid); matrix_add(B11, B22, T2, mid); strassen_recursive_local(T1, T2, P5, mid);
    matrix_sub(A12, A22, T1, mid); matrix_add(B21, B22, T2, mid); strassen_recursive_local(T1, T2, P6, mid);
    matrix_sub(A11, A21, T1, mid); matrix_add(B11, B12, T2, mid); strassen_recursive_local(T1, T2, P7, mid);

    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            int k = idx(i, j, mid);
            C[idx(i, j, n)]             = P5[k] + P4[k] - P2[k] + P6[k];
            C[idx(i, j + mid, n)]       = P1[k] + P2[k];
            C[idx(i + mid, j, n)]       = P3[k] + P4[k];
            C[idx(i + mid, j + mid, n)] = P5[k] + P1[k] - P3[k] - P7[k];
        }
    }
}


template<typename T>
Matrix<T> MPIStrassen<T>::operator*(MPIStrassen<T>& B_obj) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Datatype mpi_type = get_mpi_type<T>();

    int n = this->matrix.get_n();
    Matrix<T> C_result(n);

    int m = 1;
    while (m < n) m *= 2;

    bool distributed = (size >= 8);

    if (rank == 0) {
        std::vector<T> A_vec(n * n), B_vec(n * n);
        std::copy(this->matrix.get_data(), this->matrix.get_data() + n*n, A_vec.begin());
        std::copy(B_obj.getMatrix().get_data(), B_obj.getMatrix().get_data() + n*n, B_vec.begin());

        std::vector<T> A_use, B_use, C_use(m * m);

        if (m != n) {
            A_use.resize(m * m); B_use.resize(m * m);
            std::fill(A_use.begin(), A_use.end(), 0);
            std::fill(B_use.begin(), B_use.end(), 0);
            for(int i=0; i<n; ++i) {
                for(int j=0; j<n; ++j) {
                    A_use[i*m + j] = A_vec[i*n + j];
                    B_use[i*m + j] = B_vec[i*n + j];
                }
            }
        } else {
            A_use = A_vec; B_use = B_vec;
        }

        if (!distributed) {
            std::cout << "[WARNING] Not enough processes (need 8). Running Local." << std::endl;
            strassen_recursive_local(A_use, B_use, C_use, m);
        } else {
            int mid = m / 2;
            int q = mid * mid;

            std::vector<T> A11(q), A12(q), A21(q), A22(q);
            std::vector<T> B11(q), B12(q), B21(q), B22(q);
            
            for (int i = 0; i < mid; ++i) {
                for (int j = 0; j < mid; ++j) {
                    A11[idx(i,j,mid)] = A_use[idx(i,j,m)]; A12[idx(i,j,mid)] = A_use[idx(i,j+mid,m)];
                    A21[idx(i,j,mid)] = A_use[idx(i+mid,j,m)]; A22[idx(i,j,mid)] = A_use[idx(i+mid,j+mid,m)];
                    B11[idx(i,j,mid)] = B_use[idx(i,j,m)]; B12[idx(i,j,mid)] = B_use[idx(i,j+mid,m)];
                    B21[idx(i,j,mid)] = B_use[idx(i+mid,j,m)]; B22[idx(i,j,mid)] = B_use[idx(i+mid,j+mid,m)];
                }
            }

            std::vector<T> T1(q), T2(q);


            matrix_sub(B12, B22, T1, mid);
            MPI_Send(A11.data(), q, mpi_type, 1, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(T1.data(), q, mpi_type, 1, TAG_B_PART, MPI_COMM_WORLD);

            matrix_add(A11, A12, T1, mid);
            MPI_Send(T1.data(), q, mpi_type, 2, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(B22.data(), q, mpi_type, 2, TAG_B_PART, MPI_COMM_WORLD);

            matrix_add(A21, A22, T1, mid);
            MPI_Send(T1.data(), q, mpi_type, 3, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(B11.data(), q, mpi_type, 3, TAG_B_PART, MPI_COMM_WORLD);

            matrix_sub(B21, B11, T1, mid);
            MPI_Send(A22.data(), q, mpi_type, 4, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(T1.data(), q, mpi_type, 4, TAG_B_PART, MPI_COMM_WORLD);

            matrix_add(A11, A22, T1, mid); matrix_add(B11, B22, T2, mid);
            MPI_Send(T1.data(), q, mpi_type, 5, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(T2.data(), q, mpi_type, 5, TAG_B_PART, MPI_COMM_WORLD);

            matrix_sub(A12, A22, T1, mid); matrix_add(B21, B22, T2, mid);
            MPI_Send(T1.data(), q, mpi_type, 6, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(T2.data(), q, mpi_type, 6, TAG_B_PART, MPI_COMM_WORLD);

            matrix_sub(A11, A21, T1, mid); matrix_add(B11, B12, T2, mid);
            MPI_Send(T1.data(), q, mpi_type, 7, TAG_A_PART, MPI_COMM_WORLD);
            MPI_Send(T2.data(), q, mpi_type, 7, TAG_B_PART, MPI_COMM_WORLD);

            std::vector<T> P1(q), P2(q), P3(q), P4(q), P5(q), P6(q), P7(q);
            MPI_Recv(P1.data(), q, mpi_type, 1, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P2.data(), q, mpi_type, 2, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P3.data(), q, mpi_type, 3, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P4.data(), q, mpi_type, 4, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P5.data(), q, mpi_type, 5, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P6.data(), q, mpi_type, 6, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(P7.data(), q, mpi_type, 7, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < mid; ++i) {
                for (int j = 0; j < mid; ++j) {
                    int k = idx(i, j, mid);
                    C_use[idx(i, j, m)]             = P5[k] + P4[k] - P2[k] + P6[k];
                    C_use[idx(i, j + mid, m)]       = P1[k] + P2[k];
                    C_use[idx(i + mid, j, m)]       = P3[k] + P4[k];
                    C_use[idx(i + mid, j + mid, m)] = P5[k] + P1[k] - P3[k] - P7[k];
                }
            }
        }

        T* c_data = C_result.get_data();
        if (m != n) {
            for(int i=0; i<n; ++i) {
                for(int j=0; j<n; ++j) {
                    c_data[i*n + j] = C_use[i*m + j];
                }
            }
        } else {
             std::copy(C_use.begin(), C_use.end(), c_data);
        }
    } 
    return C_result;
}

template<typename T>
void MPIStrassen<T>::workerLoop() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype mpi_type = get_mpi_type<T>();

    if (rank >= 1 && rank <= 7) {
        
        MPI_Status status;
        MPI_Probe(0, TAG_A_PART, MPI_COMM_WORLD, &status);
        
        int count;
        MPI_Get_count(&status, mpi_type, &count);
        int q = count;
        int mid = sqrt(q);

        std::vector<T> WA(q), WB(q), WR(q);
        
        MPI_Recv(WA.data(), q, mpi_type, 0, TAG_A_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(WB.data(), q, mpi_type, 0, TAG_B_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        strassen_recursive_local(WA, WB, WR, mid);

        MPI_Send(WR.data(), q, mpi_type, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

template<> MPI_Datatype get_mpi_type<int>() { return MPI_INT; }
template<> MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }
template<> MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }
template<> MPI_Datatype get_mpi_type<long long>() { return MPI_LONG_LONG; }