#include <omp.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>

template<typename T>
class MatrixView{
public:
    T* data; // this should point to the start of the view
    int n; // this should be the size of the view
    int stride; // basically the size of the original matrix --> help w with indexing
    MatrixView(T* data, int n, int stride) : data(data), n(n), stride(stride) {}
    T& at(int i, int j){ return data[i * stride + j]; } // everything view related should be accessed through this
    const T& at(int i, int j) const { return data[i * stride + j]; }
};
// MATRIX CLASS
template<typename T>
class Matrix{
private:
    const int n;
    T* data; // 
public:
    Matrix(int n) : n(n) {
        data = new T[n * n]();
    }
    ~Matrix(){
        delete[] data;
    }
    void fillMatrix(){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                data[i * n + j] = rand() % 5;
            }
        }
    }
    void fillMatrix(T val){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                data[i * n + j] = val;
            }
        }
    }
    void printMatrix(){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                std::cout << data[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    operator MatrixView<T>() const {
        return MatrixView<T>{this->data, n, n};
    }
    operator MatrixView<T>() { return MatrixView<T>(data, n, n); }
    void add(const MatrixView<T>& A, const MatrixView<T>&B, MatrixView<T> C){
        for(int i = 0; i < A.n; i++){
            for(int j = 0; j < A.n; j++){
                C.at(i, j) = A.at(i, j) + B.at(i, j);
            }
        }
    }
    void sub(const MatrixView<T>& A, const MatrixView<T>&B, MatrixView<T> C){
        for(int i = 0; i < A.n; i++){
            for(int j = 0; j < A.n; j++){
                C.at(i, j) = A.at(i, j) - B.at(i, j);
            }
        }
    }
    void naiveMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T>& C){
        for(int i = 0; i < A.n; i++){
            for(int j = 0; j < A.n; j++){
                T sum = 0;
                for(int k = 0; k < A.n; k++){
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
            }
        }
    }
    void strassenMultiply(const MatrixView<T>& A, const MatrixView<T>& B, MatrixView<T> C){
        if (A.n <= 2){
            // TODO: either naive way or the thingy
            this->naiveMultiply(A, B, C);
            return;
        }
        // first thing first get the 8 slice of the matrix, corresponding to a..h
        // if we were to create a new matrix each time --> memory inefficient on big n
        // rather than that how about a view? -> like torch tensor
        int half = A.n / 2;
        // Split A into 4 submatrices
        MatrixView<T> a{A.data, half, A.stride};
        MatrixView<T> b{A.data + half, half, A.stride};
        MatrixView<T> c{A.data + half * A.stride, half, A.stride};
        MatrixView<T> d{A.data + half * A.stride + half, half, A.stride};

        // Split B into 4 submatrices
        MatrixView<T> e{B.data, half, B.stride};
        MatrixView<T> f{B.data + half, half, B.stride};
        MatrixView<T> g{B.data + half * B.stride, half, B.stride};
        MatrixView<T> h{B.data + half * B.stride + half, half, B.stride};

        // split C into 4 submatrices
        MatrixView<T> c11{C.data, half, C.stride};
        MatrixView<T> c12{C.data + half, half, C.stride};
        MatrixView<T> c21{C.data + half * C.stride, half, C.stride};
        MatrixView<T> c22{C.data + half * C.stride + half, half, C.stride};
        
        // compute M1​=(a+d)(e+h)
        Matrix<T> m1(half);
        #pragma omp task
        {
            Matrix<T> ad(half);
            Matrix<T> eh(half);
            add(a, d, ad);
            add(e, h, eh);
            this->strassenMultiply(ad, eh, m1);
        }
        // compute M2​=(c+d)e
        Matrix<T> m2(half);
        #pragma omp task
        {
            Matrix<T> cd(half);
            add(c, d, cd);
            this->strassenMultiply(cd, e, m2);
        }
        // compute M3​=a(f−h)
        Matrix<T> m3(half);
        #pragma omp task
        {
            Matrix<T> fh(half);
            sub(f, h, fh);
            this->strassenMultiply(a, fh, m3);
        }
        // compute M4=d(g−e)
        Matrix<T> m4(half);
        #pragma omp task
        {
            Matrix<T> ge(half);
            sub(g, e, ge);
            this->strassenMultiply(d, ge, m4);
        }
        // compute M5=(a+b)h
        Matrix<T> m5(half);
        #pragma omp task
        {
            Matrix<T> ab(half);
            add(a, b, ab);
            this->strassenMultiply(ab, h, m5);
        }
        // compute M6=(c−a)(e+f)
        Matrix<T> m6(half);
        #pragma omp task
        {
            Matrix<T> ca(half), ef(half);
            sub(c, a, ca);
            add(e, f, ef);
            this->strassenMultiply(ca, ef, m6); 
        }
        // compute M7=(b−d)(g+h)
        Matrix<T> m7(half);
        #pragma omp task
        {
            Matrix<T> bd(half), gh(half);
            sub(b, d, bd);
            add(g, h, gh);
            this->strassenMultiply(bd, gh, m7);
        }

        #pragma omp taskwait
            // C = [
            //      M1 + M4 - M5 + M7 | M3 + M5
            //      M2 +  M4 | M1 + M3 - M2 + M6
            // ]
            // C11 = M1 + M4 - M5 + M7
            // C12 = M3 + M5
            // C21 = M2 + M4
            // C22 = M1 + M3 - M2 + M6
        #pragma omp task
        {
            Matrix<T> m14(half), m145(half);
            add(m1, m4, m14);
            sub(m14, m5, m145);
            add(m145, m7, c11);
        }
        #pragma omp task
        {
            add(m3, m5, c12);
        }
        #pragma omp task
        {
            add(m2, m4, c21);
        }
        #pragma omp task
        {
            Matrix<T> m13(half), m132(half);
            add(m1, m3, m13);
            sub(m13, m2, m132);
            add(m132, m6, c22);
        }

        #pragma omp taskwait
    }
    Matrix<T> operator*(Matrix<T>& B){
        Matrix<T> C(this->n);
        #pragma omp parallel
        {
            #pragma omp single
            {
                this->strassenMultiply(*this,B,C);
            }
        }
        return C;
    }
    int get_n() const { return n; }
    T& get(int i, int j) const {return data[i * n + j];}
    Matrix<T>& operator=(const Matrix<T>& B) {
        if (this != &B) { // Check for self-assignment
            if (n != B.n) {
                throw std::invalid_argument("Cannot assign matrices of different sizes");
            }
            std::copy(B.data, B.data + n * n, this->data); // Deep copy
        }
        return *this;
    }
// private:
    
};

int main(){
    int size = 4;
    Matrix<int> A(size);
    Matrix<int> B(size);
    Matrix<int> C(size);
    
    A.fillMatrix();
    B.fillMatrix();
    C.fillMatrix(0);
    std::cout << "Matrix A:" << std::endl;
    A.printMatrix();
    std::cout << "Matrix B:" << std::endl;
    B.printMatrix();
    std::cout << "Matrix C:" << std::endl;
    C = A * B;
    
    C.printMatrix();
    return 0;
}