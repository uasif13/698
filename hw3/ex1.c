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
void init_data(double* array, int data_size);

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

    int n = 4;
    const int n_sq = n*n;
    // double window_buffer[n_sq];
    double * window_buffer;
    double *A;

    A = (double*) malloc(n_sq*sizeof(double));
    window_buffer = (double*) malloc(n_sq*sizeof(double));
    init_data(window_buffer,n_sq);
    init_data(A,n_sq);

    MPI_Win window;

    /*if (my_rank == 1) {
    	window_buffer[1] = 3.5;
    }*/
    MPI_Win_create(A, n_sq * sizeof(double),sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0,window);

    double  remote_value;
    if (my_rank == 0) {
    	MPI_Get(&remote_value,1,MPI_DOUBLE,1,1,1,MPI_DOUBLE,window);

	double my_value = 123456.7;
	MPI_Put(&my_value,1, MPI_DOUBLE,1,0,1,MPI_DOUBLE,window);
    }
    MPI_Win_fence(0,window);

    if (my_rank == 0) {
    	printf("MPI process 0: value from p1 buffer 1, %f\n", remote_value);
    }

    MPI_Win_free(&window);

    // Tell MPI to shut down.
    MPI_Finalize();

    return EXIT_SUCCESS;
}
void init_data(double* array, int data_size) {
	for (int i = 0; i< data_size; i++) {
		array[i] = 0xF &rand();
	}
}
