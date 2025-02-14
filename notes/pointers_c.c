#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

void init_data(double* data, int data_size) {
	for (int i = 0; i < data_size ; i++) {
		data[i] = rand() & 0xF;
		printf("%.1f ",data[i]);
	}
}
int main (int argc, char *argv[]) {
	double* A;
	int n = 5;
	int elem_size = sizeof(double);
	A = (double*) malloc(elem_size*n);
	

	init_data(A,n);
	printf("location of A: %d\n" ,&A);
	printf("First: %.1f \n",*A);
	A += 2;
	
	printf("Second: %.1f \n",*A);
	
}
