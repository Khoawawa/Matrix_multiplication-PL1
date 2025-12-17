#include "Matrix.h"
template <typename T> T& MatrixView<T>::at(int i, int j) { return data[i * stride + j]; }
template <typename T> const T& MatrixView<T>::at(int i, int j) const { return data[i * stride + j]; }
template<typename T> 
Matrix<T> MatrixView<T>::toMatrix()
{
    Matrix<T> ret(this->n);
    for (int i = 0; i < this->n; i++)
    {
        // iterate over every row -> each i is an x
        std::copy(
            this->data + i * this->stride,
            this->data + i * this->stride + this->n,
            ret.data + i*n
        );
    }
    return ret;
}