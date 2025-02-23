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
void mat_multi(double *A, double *B, double *C, int n, int elms_to_comm) ;
void output_matrix(double* array, int data_size);
void init_data(double* array, int data_size);
int check_result(double *C, double *D, int n);

int main(int argc, char* argv[])
{
    // Initilialise MPI and check its completion
    MPI_Init(&argc, &argv);

    // Get my rank
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // printf("%d Processes in  MPI environment.\n", comm_size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    // printf("Process %d has initialised its MPI environment.\n", my_rank);

    int flag;
    int addr_to_comm;
    double start_time, end_time, elapsed;
    int n = 2048;
    const int n_sq = n*n;
    // double window_buffer[n_sq];
    double * window_buffer;
    double *A, *B,*C, *D, *E, *F;
    int my_work = n / comm_size;
    int elms_to_comm = my_work * n;

    A = (double*) malloc(n_sq*sizeof(double));
    B = (double*) malloc(n_sq*sizeof(double));
    C = (double*) malloc(n_sq*sizeof(double));
    D = (double*) malloc(n_sq*sizeof(double));
    E = (double *) malloc(elms_to_comm*sizeof(double));
    F = (double *) malloc(elms_to_comm*sizeof(double));
    // window_buffer = (double*) malloc(n_sq*sizeof(double));
    // init_data(window_buffer,n_sq);
    if (my_rank == 0) {
	    init_data(A,n_sq);
	    printf("Initial A\n");
	    // output_matrix(A,n_sq);
	    init_data(B,n_sq);
	    printf("Initial B\n");
	    // output_matrix(B,n_sq);
    }

    start_time = MPI_Wtime();
    MPI_Win window;

    /*if (my_rank == 1) {
    	window_buffer[1] = 3.5;
    }*/
    MPI_Win_create(A, n_sq * sizeof(double),sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0,window);
MPI_Get(C,n_sq,MPI_DOUBLE,0,0,n_sq,MPI_DOUBLE,window);
    MPI_Win_fence(0,window);



	double * my_value;
	// my_value = (double *) malloc (3*sizeof(double));
	// init_data(my_value,3);
	// MPI_Put(my_value,3, MPI_DOUBLE,0,0,3,MPI_DOUBLE,window);
	MPI_Get(E,elms_to_comm,MPI_DOUBLE,0,elms_to_comm*my_rank,elms_to_comm,MPI_DOUBLE,window);
	MPI_Win_fence(0,window);

	printf("MPI process 0: output array for %d\n",my_rank );
	// output_matrix(E,elms_to_comm);

	MPI_Bcast(B,n_sq,MPI_DOUBLE,0,MPI_COMM_WORLD);
	// output_matrix(B,n_sq);

	 mat_multi(E,B,F,n,elms_to_comm);
	 printf("output matrix F\n");
	 // output_matrix(F,elms_to_comm);
	MPI_Win_fence(0,window);
	 MPI_Put(F,elms_to_comm, MPI_DOUBLE,0,elms_to_comm*my_rank,elms_to_comm,MPI_DOUBLE,window);

	 MPI_Win_fence(0,window);
	 if (my_rank == 0) {
	 
	// printf("Output Array A\n");
	// output_matrix(A,n_sq);
	end_time = MPI_Wtime();
	elapsed = end_time - start_time;
	printf("Output Array C\n");
	// output_matrix(C,n_sq);
	printf("Output Array B\n");
	// output_matrix(B,n_sq);
	mat_multi(C,B,D,n,n_sq);
	printf("Output Array D\n");
	// output_matrix(D,n_sq);
	printf("check result\n");
	flag = check_result(A,D,n);
	if (flag) printf("Test: FAILED\n");
	else {
		printf("Test: PASSED\n");
		printf("Total time %d: %f seconds.\n",my_rank,elapsed);
	}
	 }
	MPI_Win_fence(0,window);
    MPI_Win_free(&window);

    // Tell MPI to shut down.
    MPI_Finalize();

    return EXIT_SUCCESS;
}
void mat_multi(double *A, double *B, double *C, int n, int elms_to_comm) {
	int count = 0 ;
	for (int i = 0; i < elms_to_comm && count < elms_to_comm; i++) {
		for (int j = 0 ; j < n && count < elms_to_comm; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				sum += A[i*n+k] * B[n*k+j];
			}
			C[i*n+j] = sum;
			count ++;
		}
	}
}
void init_data(double* array, int data_size) {
	for (int i = 0; i< data_size; i++) {
		array[i] = 0xF &rand();
	}
}

void output_matrix(double* array, int data_size) {
	for (int i = 0; i < data_size; i++) {
		printf("%.1f ",array[i]);
	}
	printf("\n");
}
// Compare two arrays
int check_result(double *C, double *D, int n) {
	int i,j,flag = 0;
	double *cp, *dp;

	cp = C;
	dp = D;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (*cp++ != *dp++) {
				printf("ERROR: C[%d][%d] = %d != D[%d][%d]=%d\n",C[i*n+j],D[i*n+j]);
				flag = 1;
				return flag;
			}
		}
	}
	return flag;
}
