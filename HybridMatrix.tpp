#include "HybridMatrix.h"
template<typename T>
HybridMatrix<T>& HybridMatrix<T>::operator=(const Matrix<T>& B){
    this->matrix = B;
    return *this;
}
template<typename T>
MatrixView<T>* HybridMatrix<T>::splitMatrix4View(const Matrix<T>& M)
{
    int half = M.get_n() / 2;

    MatrixView<T> a11{M.get_data(), half, M.get_n()};
    MatrixView<T> a12{M.get_data() + half, half, M.get_n()};
    MatrixView<T> a21{M.get_data() + half * M.get_n(), half, M.get_n()};
    MatrixView<T> a22{M.get_data() + half * M.get_n() + half, half, M.get_n()};

    MatrixView<T>* views = new MatrixView<T>[4];
    views[0] = a11;
    views[1] = a12;
    views[2] = a21;
    views[3] = a22;
    return views;
}

template<typename T>
Matrix<T> HybridMatrix<T>::ompStrassenMult(const MatrixView<T>& A, const MatrixView<T>& B)
{
    Matrix<T> C(A.get_n());
    MatrixView<T> C_view = C.view();
#pragma omp parallel
    {
        #pragma omp single
        C.strassenMultiply(A, B, C_view);
    }
    return C;
}
template<typename T>
Matrix<T> HybridMatrix<T>::operator*(Matrix<T>& B){
    int n = this->matrix.get_n();
    Matrix<T> C(n);
    
    if (n <= 2){
        MatrixView<T> C_view = C.view();
        Matrix<T>::naiveMultiply(this->matrix, B, C_view);
        return C;
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        // setup
        int half = n / 2;
        Matrix<T> * Ms = new Matrix<T>[7]{
            Matrix<T>(half),
            Matrix<T>(half),
            Matrix<T>(half),
            Matrix<T>(half),
            Matrix<T>(half),
            Matrix<T>(half),
            Matrix<T>(half)
        };
        // start spreading tasks
        coordinator<T>(this->matrix,B,C,size-1, Ms);
        // gather results
        MatrixView<T>* C_views = HybridMatrix<T>::splitMatrix4View(C);
        //sequential again
        Matrix<T> m14(half), m145(half);
        Matrix<T>::add(Ms[0], Ms[3], m14);
        Matrix<T>::sub(m14, Ms[4], m145);
        Matrix<T>::add(m145, Ms[6], C_views[0]);

        Matrix<T>::add(Ms[2], Ms[4], C_views[1]);
        Matrix<T>::add(Ms[1], Ms[3], C_views[2]);

        Matrix<T> m13(half), m132(half);
        Matrix<T>::add(Ms[0], Ms[2], m13);
        Matrix<T>::sub(m13, Ms[1], m132);
        Matrix<T>::add(m132, Ms[5], C_views[3]);
    }
    else{
        worker<T>(rank);
    }
    return C;
}
template <typename T>
Matrix<T> HybridMatrix<T>::add_matrix(const Matrix<T>& A, const Matrix<T>& B){
    Matrix<T> C(A.get_n());
    MatrixView<T> C_view = C.view();
    Matrix<T>::add(A, B, C_view);
    return C;
}
template <typename T>
Matrix<T> HybridMatrix<T>::sub_matrix(const Matrix<T>& A, const Matrix<T>& B){
    Matrix<T> C(A.get_n());
    MatrixView<T> C_view = C.view();
    Matrix<T>::sub(A, B, C_view);
    return C;
}
template <typename T>
void worker(int rank){
    while(true){
        // ask for a task
        MPI_Send(NULL, 0, MPI_BYTE, 0, TAG_REQUEST_TASK, MPI_COMM_WORLD);
        // *checking for tag to make sure coordinator want to cont or not
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_NO_MORE_TASKS){
            // no more tasks --> terminate
            MPI_Recv(NULL, 0,MPI_BYTE, 0, TAG_NO_MORE_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout<< "worker " << rank << " is done" << std::endl;
            break;
        }
        // receive the task
        StrassenTask<T> task = HybridMatrix<T>::recvTask(0, rank);
        // do the task
        Matrix<T> C_part = HybridMatrix<T>::ompStrassenMult(task.A_part, task.B_part);
        // send the result
        int n = C_part.get_n();
        MPI_Datatype dtype = get_mpi_data_type<T>();
        MPI_Send(&task.id, 1, MPI_INT, 0, TAG_RESULT_ID, MPI_COMM_WORLD);
        MPI_Send(C_part.get_data(), n*n, dtype, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
    }
}
template <typename T>
void coordinator(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C, int num_worker, Matrix<T>* Ms){
    int half = A.get_n() / 2;
    Matrix<T>* As = A.splitQuadrantMatrix();
    Matrix<T>* Bs = B.splitQuadrantMatrix();
    // here it should send the correspond matrix to the required rank
    StrassenTask<T> tasks[7] = {
        StrassenTask<T>(1, HybridMatrix<T>::add_matrix(As[0], As[3]), HybridMatrix<T>::add_matrix(Bs[0], Bs[3])),
        StrassenTask<T>(2, HybridMatrix<T>::add_matrix(As[2], As[3]), Bs[0]),
        StrassenTask<T>(3, As[0], HybridMatrix<T>::sub_matrix(Bs[1], Bs[3])),
        StrassenTask<T>(4, As[3], HybridMatrix<T>::sub_matrix(Bs[2], Bs[0])),
        StrassenTask<T>(5, HybridMatrix<T>::add_matrix(As[0], As[1]), Bs[3]),
        StrassenTask<T>(6, HybridMatrix<T>::sub_matrix(As[2], As[0]), HybridMatrix<T>::add_matrix(Bs[0], Bs[1])),
        StrassenTask<T>(7, HybridMatrix<T>::sub_matrix(As[1], As[3]), HybridMatrix<T>::add_matrix(Bs[2], Bs[3]))
    };
    int taskSize = 7;
    std::queue<StrassenTask<T>> taskQueue;
    
    for(int i = 0; i < taskSize; i++){
        taskQueue.push(tasks[i]);
    }

    int active_worker = num_worker;
    std::vector<MPI_Request> requests;

    while(active_worker > 0)
    {
        MPI_Status status;

        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,&status);
        
        int worker = status.MPI_SOURCE;
        if (status.MPI_TAG == TAG_REQUEST_TASK){
            // receive the request to send a task
            MPI_Recv(NULL, 0,MPI_BYTE, worker, TAG_REQUEST_TASK, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            if(!taskQueue.empty()){
                // send the task if there is one
                StrassenTask<T> task = taskQueue.front();
                taskQueue.pop();
                MPI_Request reqs[2];
                HybridMatrix<T>::sendTask(task, worker, reqs);
                requests.push_back(reqs[0]);
                requests.push_back(reqs[1]);
            }
            else{
                // inform the worker to terminate
                MPI_Send(NULL, 0, MPI_BYTE, worker, TAG_NO_MORE_TASKS, MPI_COMM_WORLD);
                std:: cout << "Coordinator send stop signal to worker " << worker << std::endl;
                active_worker--;
            }
        }else if (status.MPI_TAG == TAG_RESULT_ID){
            // RECEIVE THE RESULT
            int id;
            MPI_Recv(&id, 1, MPI_INT, worker, TAG_RESULT_ID, MPI_COMM_WORLD, &status);
            MPI_Request data_request;
            MPI_Irecv(Ms[id-1].get_data(), half*half, get_mpi_data_type<T>(), worker, TAG_RESULT_DATA, MPI_COMM_WORLD, &data_request);
            requests.push_back(data_request);
        }
    }
    // wait for all requests to finish
    std::cout<< "Coordinator waiting for all results..." << std::endl;
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    std::cout << "Coordinator received all results" << std::endl;

    delete[] As;
    delete[] Bs;
}
template<typename T>
void HybridMatrix<T>::sendTask(const StrassenTask<T>& task, int dest, MPI_Request reqs[2]) {
    MPI_Datatype dtype = get_mpi_data_type<T>();
    MPI_Send(&task.id, 1, MPI_INT, dest, TAG_SEND_ID, MPI_COMM_WORLD);
    int n = task.A_part.get_n();
    MPI_Send(&n, 1, MPI_INT, dest, TAG_SEND_N, MPI_COMM_WORLD);

    MPI_Isend(task.A_part.get_data(), n*n, dtype, dest, TAG_SEND_A_DATA, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(task.B_part.get_data(), n*n, dtype, dest, TAG_SEND_B_DATA, MPI_COMM_WORLD, &reqs[1]);
}
template<typename T>
StrassenTask<T> HybridMatrix<T>::recvTask(int src, int rank) {
    MPI_Status status;
    MPI_Datatype dtype = get_mpi_data_type<T>();
    int id, n;
    MPI_Recv(&id, 1, MPI_INT, src, TAG_SEND_ID, MPI_COMM_WORLD, &status);
    MPI_Recv(&n, 1, MPI_INT, src, TAG_SEND_N, MPI_COMM_WORLD, &status);
    
    StrassenTask<T> task(id, Matrix<T>(n), Matrix<T>(n));

    MPI_Recv(task.A_part.get_data(), n*n, dtype, src, TAG_SEND_A_DATA, MPI_COMM_WORLD, &status);
    MPI_Recv(task.B_part.get_data(), n*n, dtype, src, TAG_SEND_B_DATA, MPI_COMM_WORLD, &status);
    return task;
}
template<>
MPI_Datatype get_mpi_data_type<int>(){
    return MPI_INT;
}
template<>
MPI_Datatype get_mpi_data_type<float>(){
    return MPI_FLOAT;
}
template<>
MPI_Datatype get_mpi_data_type<double>(){
    return MPI_DOUBLE;
}
template<>
MPI_Datatype get_mpi_data_type<long double>(){
    return MPI_LONG_DOUBLE;
}
template<>
MPI_Datatype get_mpi_data_type<long long>(){
    return MPI_LONG_LONG;
}