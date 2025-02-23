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
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_size);

    printf("Process %d has initialised its MPI environment.\n", comm_size);

    // Tell MPI to shut down.
    MPI_Finalize();

    return EXIT_SUCCESS;
}
