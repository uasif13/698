/*
 *Mat Mult
 CS 698 GPU - MPI + CUDA
 Spring 2025
 template for hw1-3
 Andrew Sohn
 Asif Uddin
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAXDIM 4096
#define ROOT 0

/**
 * @brief Illustrates how to initialise the MPI environment.
 **/
void mat_multi(double *A, double *B, double *C, int n, int elms_to_comm) ;
void output_matrix(double* array, int data_size);
void init_data(double* array, int data_size);
int check_result(double *C, double *D, int n);

int main(int argc, char* argv[])
{
	// Instance variables
    int comm_size;
    int my_rank;
    int flag;
    int addr_to_comm;
    double start_time, end_time, elapsed;
    int n = 2048;
    const int n_sq = n*n;
    double *A, *B,*C, *D, *E, *F;

    if (argc > 1) {
    	n = atoi(argv[1]);
	if (n > MAXDIM) {
		n = MAXDIM;
	}
    }
    // Initilialise MPI and check its completion
    MPI_Init(&argc, &argv);

    // Get my rank
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);


    int my_work = n / comm_size;
    int elms_to_comm = my_work * n;


    // malloc arrays
    A = (double*) malloc(n_sq*sizeof(double));
    B = (double*) malloc(n_sq*sizeof(double));
    C = (double*) malloc(n_sq*sizeof(double));
    D = (double*) malloc(n_sq*sizeof(double));
    E = (double *) malloc(elms_to_comm*sizeof(double));
    F = (double *) malloc(elms_to_comm*sizeof(double));

    printf("pid=%d, n=%d, elms_to_comm=%d, num_procs=%d\n", my_rank, n, elms_to_comm, comm_size);
    // init matrices
    if (my_rank == ROOT) {
	    init_data(A,n_sq);
	    printf("Initial A\n");
	    init_data(B,n_sq);
	    printf("Initial B\n");
    }

    start_time = MPI_Wtime();
    // window and duplicate
    MPI_Win window;

    MPI_Win_create(A, n_sq * sizeof(double),sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0,window);
MPI_Get(C,n_sq,MPI_DOUBLE,0,0,n_sq,MPI_DOUBLE,window);
    MPI_Win_fence(0,window);



    // Get subset of A
	double * my_value;
	MPI_Get(E,elms_to_comm,MPI_DOUBLE,0,elms_to_comm*my_rank,elms_to_comm,MPI_DOUBLE,window);
	MPI_Win_fence(0,window);

	// output_matrix(E,elms_to_comm);

	// send bcast of b
	MPI_Bcast(B,n_sq,MPI_DOUBLE,0,MPI_COMM_WORLD);
	// output_matrix(B,n_sq);

	// partial mat multi
	 mat_multi(E,B,F,n,elms_to_comm);
	 printf("output matrix F\n");
	 // output_matrix(F,elms_to_comm);
	MPI_Win_fence(0,window);

	// partial multi -> window
	 MPI_Put(F,elms_to_comm, MPI_DOUBLE,0,elms_to_comm*my_rank,elms_to_comm,MPI_DOUBLE,window);

	 MPI_Win_fence(0,window);
	 if (my_rank == ROOT) {
	 
	// printf("Output Array A\n");
	// output_matrix(A,n_sq);
	end_time = MPI_Wtime();
	elapsed = end_time - start_time;
	// output_matrix(C,n_sq);
	// output_matrix(B,n_sq);
	// local mat multi
	mat_multi(C,B,D,n,n_sq);
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
