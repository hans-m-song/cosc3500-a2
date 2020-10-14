// COSC3500, Semester 2, 2020
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <mpi.h>

// global variables to store the matrix
double *M, *M_part;
int *sendcounts, *displs;
int N, worker_count, id, total_jobs, count, leftover, worklen;
int test = 0;

int msg(int payload = 1)
{
   int data = payload;
   MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
   return data;
}

void job(double*Y, double* X)
{
   MPI_Barrier(MPI_COMM_WORLD);
   int start = displs[id];
   int end = start + sendcounts[id];
   int i, j, k;
   for (int x = start; x < end; x++) {
      i = x / N;
      j = x - start;
      k = x % N;
      Y[i] += M_part[j] * X[k];
   }
}

void sync(double* y, double* Y)
{
   MPI_Barrier(MPI_COMM_WORLD);
   for (int i = 0; i < N; i++) {
      MPI_Reduce(&y[i], &Y[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }
}

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   msg(); // signal job 
   MPI_Bcast(const_cast<double*>(X), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
   double* y = static_cast<double*>(calloc(N, N * sizeof(double)));
   job(y, const_cast<double*>(X));
   sync(y, Y);
   free(y);
}

void allocate_M()
{
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
      M[i*N + i] = std::sqrt(2.0) * randutil::randn();
      for (int j = i+1; j < N; ++j)
      {
         M[i*N + j] = M[j*N + i] = randutil::randn();
      }
   }
}

void allocate_M_part()
{
   int sum = 0;
   sendcounts = static_cast<int*>(malloc(worker_count*sizeof(int)));
   displs = static_cast<int*>(malloc(worker_count*sizeof(int)));
   for (int i = 0; i < worker_count; i++)
   {
      sendcounts[i] = count;
      if (leftover > 0)
      {
         sendcounts[i] += 1;
         leftover -= 1;
      }
      displs[i] = sum;
      sum += sendcounts[i];
   }

   M_part = static_cast<double*>(malloc(sendcounts[id]*sizeof(double)));
   MPI_Scatterv(M, sendcounts, displs, MPI_DOUBLE,
                M_part, sendcounts[id], MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

int deinit()
{
   free(M);
   free(M_part);
   free(sendcounts);
   free(displs);
   MPI_Finalize();
}

int main(int argc, char** argv)
{
   MPI_Init(&argc, &argv);
   
   MPI_Comm_size(MPI_COMM_WORLD, &worker_count);
   MPI_Comm_rank(MPI_COMM_WORLD, &id);

   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

   // get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   N = std::stoi(argv[1]);

   total_jobs = N * N;
   count = total_jobs / worker_count;
   leftover = total_jobs % worker_count;

   if (id == 0) allocate_M();

   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   allocate_M_part();

   if (id != 0)
   {
      double* X = static_cast<double*>(malloc(N*sizeof(double)));
      double* Y = static_cast<double*>(malloc(N*sizeof(double)));
      double* y = static_cast<double*>(malloc(N*sizeof(double)));
      while(msg()) {
         for (int i = 0; i < N; i++) y[i] = 0;
         MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
         job(y, X);
         sync(y, Y);
      }
      free(y);
      free(X);
      free(Y);
      deinit();
      return 0;
   }

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   msg(0); // signal completion

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   std::cout << "mpi"
      << "," << N
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
   deinit();
   return 0;
}
