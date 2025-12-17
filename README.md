# Parallel Matrix Multiplication: Naive & Strassen Algorithms

## Overview

This repository implements **matrix multiplication** using both the **Naive algorithm** and **Strassen’s algorithm**, with a strong focus on **parallel and high-performance computing techniques**. The project explores and compares different parallelization models, including:

- **OpenMP (shared-memory parallelism)**
- **MPI (distributed-memory parallelism)**
- **Hybrid OpenMP + MPI**
- **GPU acceleration using CUDA**

The goal is to analyze performance, scalability, and efficiency across multiple computing architectures.

---

## Features

- Implementation of **Naive Matrix Multiplication**
- Implementation of **Strassen Matrix Multiplication**
- Parallel execution using **OpenMP**
- Distributed execution using **MPI** for cluster-based systems
- **Hybrid OpenMP + MPI** model for cluster-based systems
- **CUDA-based GPU acceleration** for large-scale matrices
- Performance benchmarking and comparison

---

## Algorithms

### Naive Matrix Multiplication

The naive approach computes matrix multiplication with three nested loops:

- Time complexity: **O(n³)**
- Simple and easy to parallelize
- Performs well for small to medium matrix sizes

### Strassen’s Matrix Multiplication

Strassen’s algorithm reduces the number of multiplications by recursively dividing matrices:

- Time complexity: **O(n^log₂7) ≈ O(n².81)**
- More efficient for large matrices
- Higher overhead and memory usage
- Requires careful optimization in parallel environments

---

## Project Structure

```text
Matrix_multiplication-PL1/
├── include/         
│   ├── Matrix.tpp
│   ├── Matrix.h
│   ├── MatrixView.tpp
│   ├── HybridMatrix.h
│   ├── HybridMatrix.tpp
│   └── strassen.h
├── sequential.cpp
├── omp.cpp
├── MPIStrassen.tpp
├── MPIStrassen.h
├── main_MPI.cpp
├── main_hybrid.cpp
├── gpu.cu
└── README.md
```

---

##  Compilation and Execution

### Sequential

```bash
g++ -fopenmp sequential.cpp -Iinclude -o seq
./seq
```

### OpenMP

```bash
g++ -fopenmp omp.cpp -Iinclude -o omp
OMP_NUM_THREADS=16 ./omp
```

### MPI (Run on the provided cluster)

```bash
mpicxx -fopenmp -Iinclude main_MPI.cpp -o main_MPI
mpirun -np 24 --hostfile host.txt ./main_MPI
```

### Hybrid OpenMP + MPI (Run on the provided cluster)

```bash
mpicxx -fopenmp -Iinclude main_hybrid.cpp -o main_hybrid
OMP_NUM_THREADS=8 mpirun -np 24 --hostfile host.txt ./main_hybrid
```

### CUDA

```bash
nvcc -O3 -arch=sm_70 gpu.cu -o gpu
./gpu
```

---

## 📊 Performance Evaluation

- Execution time measured for different matrix sizes
- Comparison between Naive and Strassen algorithms
- Speedup analysis across OpenMP, MPI, Hybrid, and CUDA implementations
- Evaluation of scalability and communication overhead

---


