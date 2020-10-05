// COSC3500, Semester 2, 2020
// Assignment 2
// Header file for the eigensolver functions.
// You need to implement the function MatrixVectorMultiply() yourself, and it is called by
// the eigenvalues_arpack function.
// Requires linking against libarpack

#if !defined(COSC2500_EIGENSOLVER_H)
#define COSC2500_EIGENSOLVER_H

#include <chrono>
#include <vector>

// The external function to evaluate the matrix operation y = Matrix * x
void MatrixVectorMultiply(double* Y, const double* X);

struct EigensolverInfo
{
   int NumMultiplies;
   std::chrono::microseconds TimeInEigensolver;
   std::chrono::microseconds TimeInMultiply;
   std::vector<double> Eigenvalues;
};

// Obtains the nev largest magnitude eigenvectors of a n*n matrix, to a specified tolerance
EigensolverInfo eigenvalues_arpack(int n, int nev, double tol = 1E-14);

#endif
