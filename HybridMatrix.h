#ifndef HYBRID_MATRIX_H
#define HYBRID_MATRIX_H
#include "Matrix.h"
template<typename T>
class HybridMatrix : public Matrix<T>{
public:
    HybridMatrix(int n) : Matrix<T>(n) {}
    
}


#endif // HYBRID_MATRIX_H