#include <iostream>
#include <cmath>
#include <chrono>

// Addition of arrays using a stride loop

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

__global__
void add(int n, double* x, double const* y)
{
   int index = threadIdx.x;
   int stride = blockDim.x;
   for (int i = index; i < n; i += stride)
   {
      x[i] = x[i] + y[i];
   }
}

int main()
{
   int N = 1<<20; // pow(2,20) = 1,048,576

   // allocate memory
   double* x;
   checkError(cudaMallocManaged(&x, N*sizeof(double)));

   double* y;
   checkError(cudaMallocManaged(&y, N*sizeof(double)));

   // initialize arrays
   for (int i = 0; i < N; i++)
   {
      x[i] = 1.0;
      y[i] = 2.0;
   }

   auto t1 = std::chrono::high_resolution_clock::now();

   add<<<1, 1>>>(N, x, y);
   checkError(cudaDeviceSynchronize());

   auto t2 = std::chrono::high_resolution_clock::now();

   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

   std::cout << "Time = " << duration << " us\n";

   // clean up
   cudaFree(x);
   cudaFree(y);
}
