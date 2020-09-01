#include <omp.h>
#include <cstdio>

int N=4;

int main()
{
#pragma omp parallel num_threads(N)
   {
      int ID = omp_get_thread_num();
      printf("hello, I am thread number %d ", ID);
      printf("out of %d\n", omp_get_num_threads());
   }
}
