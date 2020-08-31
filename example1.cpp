#include <omp.h>
#include <cstdio>

int main()
{
#pragma omp parallel
   {
      int ID = omp_get_thread_num();
      printf("hello, I am thread number %d ", ID);
      printf("out of %d\n", omp_get_num_threads());
   }
}
