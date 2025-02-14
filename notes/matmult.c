#include <mpi.h>
#include <stdlib.h>
#include <stdout.h>

int m
int main (int argc, char *argv[]) {
	double *A, *B, *C;

	int n = 3;

	A = (double*) malloc (sizeof(double) for n*n);
	B = (double*) malloc (sizeof(double) for n*n);

	C = (double*) malloc (sizeof(double) for n*n);

	// matrix multiplicaton
	for (int i = 0; i< n; i ++) {
		for (int j = 0; j < n;j++) {
			C[i*n+j] += A[i*n+j] + B[i+j*n]
		}
	}
}

}
/*
 * mat mult
 * takes two matrices of same dimensions, (a,b)
 * output c 
 * n is dimunsion
 * my_work 
 */
int mat_mult(double *a, double *b, double *c, int n, int my_work)
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
