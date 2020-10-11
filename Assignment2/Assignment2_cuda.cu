// COSC3500, Semester 2, 2020
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>

// global variables to store the matrix
double* M = nullptr;
double* d_M;
int N = 0;

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << ": " 
                << cudaGetErrorString(e) << "\n";
      abort();
   }
}

__global__
void job(double* Y, double* X, double* M, int N)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x_stride = blockDim.x * gridDim.x;
   int y_stride = blockDim.y * gridDim.y;

   int y;
   for (int i = x; i < N; i += x_stride)
   {
      y = 0;
      for (int j = y; j < N; j += y_stride)
      {
         y += M[i*N+j] * X[j];
      }
      Y[i] = y;
   }
}

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   double *d_Y, *d_X;
   int size = N * sizeof(double);
   int block_size=32;
   int grid_size=((N + block_size - 1) / block_size);

   checkError(cudaMalloc((void**)&d_Y, size));
   checkError(cudaMalloc((void**)&d_X, size));

   cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);

   job<<<grid_size, block_size>>>(d_Y, d_X, d_M, N);

   //checkError(cudaDeviceSynchronize());

   checkError(cudaMemcpy((void*)Y, d_Y, size, cudaMemcpyDeviceToHost));
   checkError(cudaMemcpy((void*)X, d_X, size, cudaMemcpyDeviceToHost));

   cudaFree(d_Y);
   cudaFree(d_X);
}

int main(int argc, char** argv)
{
   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

   // get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   N = std::stoi(argv[1]);
   
   // Allocate memory for the matrix
   int size = N * N * sizeof(double);
   M = static_cast<double*>(malloc(size));
   checkError(cudaMalloc(&d_M, size));

   // seed the random number generator to a known state
   randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < N; ++i)
   {
      M[i*N] = std::sqrt(2.0) * randutil::randn();
      for (int j = i+1; j < N; ++j)
      {
         M[i*N + j] = M[j*N + i] = randutil::randn();
      }
   }
   checkError(cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice));

   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   std::cout << "Obtained " << Info.Eigenvalues.size() << " eigenvalues.\n";
   std::cout << "The largest eigenvalue is: " << std::setw(16) << std::setprecision(12) << Info.Eigenvalues.back() << '\n';
   std::cout << "Total time:                             " << std::setw(12) << TotalTime.count() << " us\n";
   std::cout << "Time spent in initialization:           " << std::setw(12) << InitializationTime.count() << " us\n";
   std::cout << "Time spent in eigensolver:              " << std::setw(12) << Info.TimeInEigensolver.count() << " us\n";
   std::cout << "   Of which the multiply function used: " << std::setw(12) << Info.TimeInMultiply.count() << " us\n";
   std::cout << "   And the eigensolver library used:    " << std::setw(12) << (Info.TimeInEigensolver - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Total serial (initialization + solver): " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
   std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

   // free memory
   free(M);
   cudaFree(d_M);
}
