#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <stdbool.h>

void my_bcast(void *data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator)
{
  int world_rank;
  MPI_Comm_rank(communicator, &world_rank);
  int world_size;
  MPI_Comm_size(communicator, &world_size);
  int comm_rank = (world_rank - root + world_size) % world_size;
  for (int i = 1; i < world_size; i *= 2)
  {
    if (comm_rank < i)
    {
      int target_rank = (comm_rank + i + root + world_size) % world_size;
      MPI_Send(data, count, datatype, target_rank, 0, communicator);
    }
    else if (comm_rank >= i && comm_rank < 2 * i)
    {
      int target_rank = (comm_rank - i + root + world_size) % world_size;
      MPI_Recv(data, count, datatype, target_rank, 0, communicator, MPI_STATUS_IGNORE);
    }
  }
}

int main(int argc, char **argv)
{
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int data;
  if (world_rank == 0)
  {
    data = 100;
    printf("Process 0 broadcasting data %d\n", data);
    my_bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  else
  {
    my_bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received data %d from root process\n", world_rank, data);
  }

  MPI_Finalize();
}
