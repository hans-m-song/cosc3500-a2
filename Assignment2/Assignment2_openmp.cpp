// COSC3500, Semester 2, 2020
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <omp.h>

// global variables to store the matrix

double* M = nullptr;
int N = 0;

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   #pragma omp parallel for default(none), shared(M, N, Y, X)
   for (int i = 0; i < N; ++i)
   {
      #pragma critical
      Y[i] = 0;
      for (int j = 0; j < N; ++j)
      {
         #pragma critical
         Y[i] += M[i*N+j] * X[j];
      }
   }
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
   M = static_cast<double*>(malloc(N*N*sizeof(double)));

   // seed the random number generator to a known state
   randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < N; ++i)
   {
      M[i*N+i] = std::sqrt(2.0) * randutil::randn();
      for (int j = i+1; j < N; ++j)
      {
         M[i*N + j] = M[j*N + i] = randutil::randn();
      }
   }
   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   std::cout << "mpi"
      << "," << Info.Eigenvalues.size()
      << "," << Info.Eigenvalues.back()
      << "," << TotalTime.count()
      << "," << InitializationTime.count()
      << "," << Info.TimeInEigensolver.count()
      << "," << Info.TimeInMultiply.count()
      << "," << (Info.TimeInEigensolver - Info.TimeInMultiply).count()
      << "," << (TotalTime - Info.TimeInMultiply).count()
      << "," << Info.NumMultiplies
      << "," << (Info.TimeInMultiply / Info.NumMultiplies).count()
      << "\n";

   // free memory
   free(M);
}
