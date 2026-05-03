#include<omp.h>
#define N 1000
#define CHUNKSIZE 100
#include <iostream>

int main(int argc, char *argv[]) {
    int i, chunk;
    float a[N], b[N], c[N];

    for (i = 0; i < N; i++)
        a[i] = b[i] = i * 1.0;
    chunk = CHUNKSIZE;

    #pragma omp parallel shared(a,b,c,chunk) private(i)
    {
        #pragma omp for schedule(static, chunk) nowait
        for (i = 0; i < N; i++){
            if (omp_get_thread_num() == 3)
                std::cout<<"Thread "<<omp_get_thread_num()<<" processing index "<<i<<"\n";
            c[i] = a[i] + b[i];
        }
    }

    std::cout << "Suma de vectores completada\n";

    return 0;
}