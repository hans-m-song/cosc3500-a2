#include <mpi.h>
#include <cstdio>

using namespace std;

int main(int argc, char** argv)
{
   MPI_Init(&argc, &argv);

   // maximum number of messages to send
   int count_limit = 4;

   // Find out rank, size
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   int my_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   // rank 0 starts off
   if (my_rank == 0)
   {
      int data = 66;
      MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      printf("Process 0 sent number %d to process 1\n", data);
   }

   for (int count = 0; count < count_limit; ++count)
   {
      int data;
      int from = (my_rank-1+world_size) % world_size;
      MPI_Recv(&data, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Process %d received number %d from process %d\n", my_rank, data, from);

      if (count < count_limit-1 || my_rank != 0)
      {
         int send_to = (my_rank+1) % world_size;
         data = (data*37) % 100;
         MPI_Send(&data, 1, MPI_INT, send_to, 0, MPI_COMM_WORLD);
         printf("Process %d sent number %d to process %d\n", my_rank, data, send_to);
      }
   }

   MPI_Finalize();
}
