#include <omp.h>
#include <cstdio>

using namespace std;

int x = 0;
int y = 0;

void thread1()
{
   y = x+1;
   printf("y=%d\n", y);
}

void thread2()
{
   x = y+1;
   printf("x=%d\n", x);
}

int main()
{
#pragma omp parallel num_threads(2)
   {
      if (omp_get_thread_num() == 0)
         thread1();
      else
         thread2();
   }
}
