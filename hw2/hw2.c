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

int mat_mult(double *A, double *B, double *C, int n, int n_local);
void init_data(double* data, int data_size);
void output_matrix(double* data, int data_size);
void create_send_matrix(double* recv_matrix,double*orig_matrix,int elms_to_comm, int pid);
int check_result(double*C, double*D, int n);

int main(int argc, char *argv[]) {
  int i, n = 8,n_sq, flag, my_work;
  int my_rank, num_procs = 4;
  double *A, *B, *C, *D, *E, *F;	/* D is for local computation, E is for buffers */
  int addr_to_comm, elms_to_comm;
  double start_time, end_time, elapsed;

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
  C = (double *) malloc(sizeof(double) * n_sq);
  D = (double *) malloc(sizeof(double) * n_sq);
  E = (double *) malloc(sizeof(double) * n_sq);
  F = (double *) malloc(sizeof(double) * n_sq);

  if (my_rank == ROOT) {
    printf("pid=%d: num_procs=%d n=%d my_work=%d\n",\
	   my_rank, num_procs, n, my_work);

	  init_data(A,n_sq);
	  init_data(B,n_sq);
	  printf("initial A\n");
	  output_matrix(A,n_sq);
	  printf("\n Initial B\n");
	  output_matrix(B,n_sq);
	  printf("\n");
  }
  
  start_time = MPI_Wtime();
  printf("pid: %d\n", my_rank);
  printf("Scatter process\n");
  
  MPI_Scatter(A, elms_to_comm, MPI_DOUBLE, E,elms_to_comm,MPI_DOUBLE,0, MPI_COMM_WORLD);
  output_matrix(E,elms_to_comm);
  printf("\n");
  
  /* Broadcast B to every one */
  MPI_Bcast(B, n_sq, MPI_DOUBLE, 0, MPI_COMM_WORLD);


  /* Each process computes its own mat mult */
  
  printf("Matrix Multiplication");
  printf("pid: %d  mat Mult\n",my_rank);
  mat_mult(E, B, F, n, elms_to_comm);
  output_matrix(F,elms_to_comm);
  
  output_matrix(F,elms_to_comm);
    /* Receive my_work rows of C from num_procs - 1 processes */
  
  printf("MPI_Gather \n");
  printf("pid: %d  \n",my_rank);

  
  	MPI_Gather(F,elms_to_comm, MPI_DOUBLE,C, elms_to_comm,MPI_DOUBLE,0, MPI_COMM_WORLD);

  if (my_rank == ROOT) {
    end_time = MPI_Wtime();
    elapsed = end_time - start_time;

    /* Local computation for comparison: results in D */
  output_matrix(C,n_sq);
    mat_mult(A, B, D, n, elms_to_comm);

    printf("check result\n");
  output_matrix(C,n_sq);
    flag = check_result(C,D,n);
    if (flag) printf("Test: FAILED\n");
    else {
      printf("Test: PASSED\n");
      printf("Total time %d: %f seconds.\n", my_rank, elapsed);
    }
  }
  printf ("free heap pointers");
  MPI_Finalize();
  return 0;
}
void output_matrix(double *data, int datasize) {
	for (int i = 0; i < datasize; i++) {
		printf("%.1f ",data[i]);
	}
}

int mat_mult(double *a, double *b, double *c, int n, int my_work) {
  int i, j, k, sum=0;
  for (i=0; i<my_work; i++) {
    for (j=0; j<n; j++) {
      sum=0;
      for (k=0; k<n; k++)
	sum = sum + a[i*n + k] * b[k*n + j];
      c[i*n + j] = sum;
    }
  }
  return 0;
}

/* Initialize an array with random data */
void init_data(double *data, int data_size) {
  for (int i = 0; i < data_size; i++)
    data[i] = rand() & 0xf;
}
void create_send_matrix(double* recv_matrix, double*orig_matrix, int elms_to_comm, int my_rank) {
	for (int i = elms_to_comm*my_rank;i<elms_to_comm*(my_rank+1);i++) {
		recv_matrix[i] = orig_matrix[i];
	}
}

/* Compare two matrices C and D */
int check_result(double *C, double *D, int n){
  int i,j,flag=0;
  double *cp,*dp;

  cp = C;
  dp = D;

  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      if (*cp++ != *dp++) {
	printf("ERROR: C[%d][%d]=%d != D[%d][%d]=%d\n",C[i*n + j],D[i*n + j]);
	flag = 1;
	return flag;
      }
    }
  }
  return flag;
}

/*
  End of file: template.c
 */
