#include <mpi.h>
#include <cstdio>
using namespace std;
int main(int argc, char** argv)
{
   MPI_Init(&argc, &argv);

   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   printf("Hello from process %d out of %d\n", my_rank, world_size);

   MPI_Finalize();
}
