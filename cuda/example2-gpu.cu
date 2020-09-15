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
   int index = blockIdx.x*blockDim.x + threadIdx.x;
   int stride = blockDim.x*gridDim.x;
   for (int i = index; i < n; i += stride)
   {
      x[i] = x[i] + y[i];
   }
}

int main()
{
   int N = 1<<20; // pow(2,20) = 1,048,576

   // allocate memory on the host
   double* x = new double[N];
   double* y = new double[N];

   // initialize arrays
   for (int i = 0; i < N; i++)
   {
      x[i] = 1.0;
      y[i] = 2.0;
   }

   // allocate memory on the device
   double* xDevice;
   double* yDevice;
   checkError(cudaMalloc(&xDevice, N*sizeof(double)));
   checkError(cudaMalloc(&yDevice, N*sizeof(double)));

   // copy memory from host to device
   checkError(cudaMemcpy(xDevice, x, N*sizeof(double), cudaMemcpyHostToDevice));
   checkError(cudaMemcpy(yDevice, y, N*sizeof(double), cudaMemcpyHostToDevice));

   int Threads = 256;
   int Blocks = (N+Threads-1)/Threads;

   auto t1 = std::chrono::high_resolution_clock::now();

   add<<<Blocks, Threads>>>(N, xDevice, yDevice);
   checkError(cudaDeviceSynchronize());

   auto t2 = std::chrono::high_resolution_clock::now();

   // copy memory from device back to host
   checkError(cudaMemcpy(x, xDevice, N*sizeof(double), cudaMemcpyDeviceToHost));

   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

   std::cout << "Time = " << duration << " us\n";

   // clean up
   cudaFree(x);
   cudaFree(y);
}
