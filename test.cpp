#include "Matrix.h"
#include <iostream>
int main(){
    Matrix<int> A(8);
    A.fillMatrix();
    std::cout << "Matrix A: " << std::endl;
    A.printMatrix();
    Matrix<int>* split = A.splitQuadrantMatrix();
    for (int i = 0; i < 4; i++){
        std::cout << "Matrix " << i << ": " << std::endl;
        split[i].printMatrix();
    }
    return 0;
}