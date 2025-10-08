#include<omp.h>
#include <stdio.h>

int main(){
    #pragma omp parallel
    {
        printf("Hello from thread %d, in parallel region\n", omp_get_thread_num());
    }
    return 0;
}