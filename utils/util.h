#ifndef UTIL_H
#define UTIL_H

#include <omp.h>
#include <vector>

// Allocate a square matrix of size n x n
double** allocate_matrix(int n);

// Deallocate a matrix
void deallocate_matrix(double** mat, int n);

// Parallel matrix addition: C = A + B
void matrix_add(double** A, double** B, double** C, int n);

// Parallel matrix subtraction: C = A - B
void matrix_sub(double** A, double** B, double** C, int n);

// Print matrix (for debugging)
void print_matrix(double** mat, int n);

#endif
