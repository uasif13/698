/*
 * Mat Mult
 * CS698 GPU cluster programming - MPI + CUDA 
 * Spring 2025
 * template for HW1 - 3
 * HW1 - point to point communication
 * HW2 - collective communication
 * HW3 - one-sided communication
 * Andrew Sohn
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define COLOR 1<<10
#define MAXDIM 1<<12		/* 4096 */
#define ROOT 0

void init_data(double* data, int data_size);
void output_matrix(double* data, int data_size);

int main(int argc, char *argv[]) {
  int i, n = 4,n_sq, flag, my_work;
  int my_rank, num_procs = 2;
  double *A, *B;	/* D is for local computation, E is for buffers */
  int addr_to_comm, elms_to_comm;

  MPI_Comm world = MPI_COMM_WORLD;
  
  MPI_Init(&argc, &argv);
  

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
  if (argc > 1) {
    n = atoi(argv[1]);
    if (n>MAXDIM) n = MAXDIM;
  }
  n_sq = n * n;
  
  my_work  = n/num_procs;
  elms_to_comm = my_work * n;
  
  A = (double *) malloc(sizeof(double) * n_sq);
  B = (double *) malloc(sizeof(double) * n_sq);

  if (my_rank == ROOT) {
    printf("pid=%d: num_procs=%d n=%d my_work=%d\n",\
	   my_rank, num_procs, n, my_work);

	  init_data(A,n_sq);
	  printf("Initial matrix");
  	output_matrix(A,n_sq);
  } 	  
  double * window_buffer;
  MPI_Win window; 
  MPI_Win_create(&A, (MPI_Aint)n_sq*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
  MPI_Win_fence(0,window);
  printf("Window created");
  
  // window

  if (my_rank != ROOT) {
  
    printf("pid=%d: num_procs=%d n=%d elms_to_comm=%d\n",my_rank, num_procs, n, elms_to_comm); 
 printf("Matrix B output rank 1\n");
 MPI_Get(B,n_sq,MPI_DOUBLE,0,0, n_sq, MPI_DOUBLE, window);
 MPI_Win_fence(0,window);
 
 output_matrix(B,n_sq);
  } 
 
 MPI_Win_fence(0,window);
  MPI_Win_free(&window);
  MPI_Finalize();
  return 0;
}
void output_matrix(double *data, int datasize) {
	for (int i = 0; i < datasize; i++) {
		printf("%.1f ",data[i]);
	}
}

/* Initialize an array with random data */
void init_data(double *data, int data_size) {
  for (int i = 0; i < data_size; i++)
    data[i] = rand() & 0xf;
}

/*
  End of file: template.c
 */
