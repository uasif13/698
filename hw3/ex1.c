/**
 * @author RookieHPC
 * @brief Original source code at https://rookiehpc.org/mpi/docs/mpi_init/index.html
 **/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * @brief Illustrates how to initialise the MPI environment.
 **/
int main(int argc, char* argv[])
{
    // Initilialise MPI and check its completion
    MPI_Init(&argc, &argv);

    // Get my rank
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    printf("%d Processes in  MPI environment.\n", comm_size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    printf("Process %d has initialised its MPI environment.\n", my_rank);
    // Tell MPI to shut down.

    const int ARRAY_SIZE = 2;
    double window_buffer[ARRAY_SIZE];
    MPI_Win window;

    if (my_rank == 1) {
    	window_buffer[1] = 3.5;
    }
    MPI_Win_create(window_buffer, ARRAY_SIZE * sizeof(int),sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

    MPI_Win_fence(0,window);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
