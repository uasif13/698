#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

void init_data(double* data, int datasize);

int main(int argc, char *argv[]) {
	// instance variables
	int n = 32;
	int n_sq = n*n;
	int my_rank;
	int num_procs = 6;
	int my_work;
	int elms_to_comm;
	int flag=1;
	double *A;
	
	// mpi initialization
	MPI_Comm world = MPI_COMM_WORLD;
	
	MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	// initialize A
	A = (double *) malloc(sizeof(double)*n_sq);

	init_data(A,n_sq);
	
	my_work = n/num_procs;
	elms_to_comm = my_work *n;
	printf("INFO: rank: %d, n: %d, my_work: %d, elms_to_comm: %d, num_procs: %d\n", my_rank, n, my_work, elms_to_comm, num_procs);

	if (my_rank == 0) {
		printf("Send code: ");
		// send A to num_procs -1 proccesses
		for (int i = 0; i < num_procs; i++) {
			for( int j = 0;j < n; j++) {
				printf("Entered send loop %d times\n", j+i*n);

				printf("Send: Value: %f, proc: %d, elem: %d ",&A, i, j);
				MPI_Send(&A, elms_to_comm, MPI_DOUBLE,my_rank, flag, MPI_COMM_WORLD);
				A++;
				printf("Send: %f \n",&A);
			} 
		}
	} else {
		// read it
		for (int m = 0; m < num_procs; m++) {
			for (int k = 0; k < n; k++) {
				printf("Entered recv loop %d times\n", k+m*n);
				printf("Recv: Value: %f, proc: %d, elem: %d \n",&A, m, k);
				MPI_Recv(&A, elms_to_comm, MPI_DOUBLE, 0, flag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				A++;
				printf("Recv: %f \n", &A);
			}
		}
	}
	MPI_Finalize();

}

void init_data(double * data, int data_size) {
	for (int i = 0; i < data_size;i++) {
	
		data[i] = rand()&0xf;
		printf("%.1f ",data[i]);
	}
		
}
