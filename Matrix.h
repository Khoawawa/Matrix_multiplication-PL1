#ifndef MATRIX_H
#define MATRIX_H
#include <omp.h>
#include <iostream>
#include <algorithm>
template<typename T>
class Matrix;
template<typename T>
class MatrixView{
public:
    T* data; // this should point to the start of the view
    int n; // this should be the size of the view
    int stride; // basically the size of the original matrix --> help w with indexing
    MatrixView(): data(nullptr), n(0), stride(0) {};
    MatrixView(T* data, int n, int stride) : data(data), n(n), stride(stride) {}
    T& at(int i, int j); // everything view related should be accessed through this
    const T& at(int i, int j) const;
    Matrix<T> toMatrix();
    int get_n() const { return n; }
};

template<typename T>
class Matrix{
    friend class MatrixView<T>; 
private:
    int n;
    T* data; // 
    
public:
    T* get_data() const { return data; }
    Matrix() : n(0), data(nullptr) {}
    Matrix(int n) : n(n) {
        data = new T[n * n]();
    }
    Matrix(const Matrix& other) : n(other.n) {
        data = new int[n * n];
        std::copy(other.data, other.data + n*n, data);
    }
    ~Matrix(){
        delete[] data;
    }
    void fillMatrix();
    void fillMatrix(T val);
    void printMatrix() const;
    operator MatrixView<T>() const;
    operator MatrixView<T>();
    static void add(const MatrixView<T>& A, const MatrixView<T>&B, MatrixView<T> C);
    static void sub(const MatrixView<T>& A, const MatrixView<T>&B, MatrixView<T> C);
    static void naiveMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C);
    MatrixView<T> view();
    virtual void strassenMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C);
    virtual Matrix<T> operator*(Matrix<T>& B);
    int get_n() const;
    T& get(int i, int j) const;
    Matrix<T>& operator=(const Matrix<T>& B);
    Matrix<T>* splitQuadrantMatrix();
};
#include "Matrix.tpp"
#include "MatrixView.tpp"
#endif