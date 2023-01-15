#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

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
  if (argc != 3)
  {
    fprintf(stderr, "Usage: compare_bcast num_elements num_trials\n");
    exit(1);
  }

  int num_elements = atoi(argv[1]);
  int num_trials = atoi(argv[2]);

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  double total_my_bcast_time = 0.0;
  double total_mpi_bcast_time = 0.0;
  int i;
  int *data = (int *)malloc(sizeof(int) * num_elements);
  assert(data != NULL);

  for (i = 0; i < num_trials; i++)
  {
    // Time my_bcast
    // Synchronize before starting timing
    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time -= MPI_Wtime();
    my_bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    // Synchronize again before obtaining final time
    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time += MPI_Wtime();

    // Time MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time -= MPI_Wtime();
    MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time += MPI_Wtime();
  }

  // Print off timing information
  if (world_rank == 0)
  {
    printf("Data size = %d, Trials = %d\n", num_elements * (int)sizeof(int),
           num_trials);
    printf("Avg my_bcast time = %lf\n", total_my_bcast_time / num_trials);
    printf("Avg MPI_Bcast time = %lf\n", total_mpi_bcast_time / num_trials);
  }

  free(data);
  MPI_Finalize();
}
