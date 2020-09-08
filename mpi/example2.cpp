#include <mpi.h>
#include <cstdio>

using namespace std;

int main(int argc, char** argv)
{
   MPI_Init(&argc, &argv);

   // Find out rank, size
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   if (my_rank == 0)
   {
      int number = 66;
      MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      printf("Process 0 sent number %d to process 1\n", number);
   }
   else if (my_rank == 1)
   {
      int number;
      MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Process 1 received number %d from process 0\n", number);
   }

   MPI_Finalize();
}
