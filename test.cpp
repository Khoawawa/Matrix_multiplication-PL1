#include <mpi.h>
#include <iostream>
#include <queue>
#include <utility> // for std::pair

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int TAG_ID = 1;
    const int TAG_DATA = 2;

    if (rank == 0) {
        std::queue<std::pair<int, int>> messageQueue; // (id, value)
        int total_msgs = (size - 1) * 2; // each worker sends 2 messages
        MPI_Status status;

        std::cout << "Coordinator waiting for " << total_msgs << " messages..." << std::endl;

        for (int i = 0; i < total_msgs; ++i) {
            int msg;
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            messageQueue.push({status.MPI_TAG, msg});

            std::cout << "[Recv] From worker " << status.MPI_SOURCE
                      << " | Tag: " << status.MPI_TAG
                      << " | Value: " << msg << std::endl;
        }

        std::cout << "\n=== Final queue order ===" << std::endl;
        int idx = 1;
        while (!messageQueue.empty()) {
            auto [tag, value] = messageQueue.front();
            messageQueue.pop();
            std::cout << idx++ << ". Tag=" << tag << " Value=" << value << std::endl;
        }

    } else {
        // Each worker sends 2 messages
        int id = rank;
        int data1 = rank * 10 + 1;
        int data2 = rank * 10 + 2;

        // Send "id" first, then "data"
        MPI_Send(&id, 1, MPI_INT, 0, TAG_ID, MPI_COMM_WORLD);
        MPI_Send(&data1, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD);

        // Send another pair after small delay to simulate overlap
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Send(&id, 1, MPI_INT, 0, TAG_ID, MPI_COMM_WORLD);
        MPI_Send(&data2, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD);

        std::cout << "Worker " << rank << " sent both messages.\n";
    }

    MPI_Finalize();
    return 0;
}