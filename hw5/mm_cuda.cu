/*
  Andrew Sohn
  10/27/2025
  CS698 MPI+CUDA Programming
  *
  The MPI+CUDA program compiles and passes the test because they are all zeros.
  Fill the functions.

  NOTE:
  need to place nvidia Common directory two dirs above the current dir
  or 
  change the Makefile reference of Common
*/
#include <iostream>
using std::cerr;
using std::endl;

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <unistd.h>
#include <sys/time.h>

#define TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define THREADS_PER_BLOCK 256

#define MAX_BUF_SIZE 1<<25
extern "C" {
  int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim );
}
void output_vector(int* data, int datasize);
int matrix_multiply_cpu(int my_rank,int *a, int *b, int *c, int n, int my_work) {
    int i, j, k, sum = 0;
    for (i = 0; i < my_work*n; i ++){
    	sum = 0;
	j = (i/n)*n;
	k = i%n;

	while (j < (i/n)*n+n && k < n*n){
	      sum += a[j]*b[k];
	      j ++;
	      k = k + n;
	}
	c[i] = sum;
    }

  return 0;
}

int compare_cpu(int my_rank, int *host, int *dev, int n, int my_work) {
  int i,j,idx;

  for (i=0; i<my_work; i++) {
    for (j=0; j<n; j++) {
      idx = i*my_work + j;
      if (dev[idx] != host[idx]) {
	printf("DIFFERENT: rank=%d: dev[%d][%d]=%d != host[%d][%d]=%d\n", \
	       my_rank,i,j,dev[idx],i,j,host[idx]);
	return 0;
      }
    }
  }

  return 1;
}

__global__ void mat_mult_cuda(int my_rank, int a_width,int my_work, int *d_a, int *d_b, int *d_c, int tile_width){
  /* 
  __shared__ int a_shared[][] ...
  __shared__ int b_shared[][] ...
 */
	extern	__shared__ int a_shared[];
	extern __shared__ int b_shared[];
	int shift = my_rank*my_work;
	int n_sq = a_width*a_width;
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
			
	int index = i+j*a_width;

	// does not include zeros
	if (shift/a_width+j < a_width)
	   for (int n = j*a_width ; n < (j+1)*a_width; n++){
		a_shared[n] = d_a[n];
		}
	    	 
	


	// does not include zeros
	for (int k = i; k < n_sq; k=k+a_width)
	    b_shared[my_work+k] = d_b[k];

	printf("%d %d: %d\n", i, j, index);	
	if (i == 0 && j == 0){
	   printf("length of a: %d\n",my_work);
	   for (int m = 0; m < a_width; m++)
	          printf("a_shared: %d ",a_shared[m]);
	   }

	
	__syncthreads();
	if (i == 0 && j == 0){
	   for (int m = 0; m < a_width * a_width; m=m+a_width)
	       printf("b_shared: %d ",b_shared[m]);
	   }

	__syncthreads();
	int sum = 0;
	for (int l = 0; l < a_width; l++){
	    int a_index = index/a_width*a_width+l;
	    int b_index = index%a_width+l*a_width;
	    if (shift+a_index < n_sq && b_index < n_sq)
	       sum = sum + a_shared[a_index]*b_shared[my_work+b_index];
	    if (i == 0 && j == 0) {
	        printf("%d %d %d: %d\n", index, a_index, b_index, sum);
	    }

	     
	  
	}
	d_c[index] = sum;
	}

void print_lst_cpu(int name,int rank,int n, int *l){
  int i=0;
  printf("CPU rank=%d: %d: ",rank,name);
  for (i=0; i<n; i++) printf("%x ",l[i]);
  printf("\n");
}

int matrix_multiply_cuda(int nprocs, int my_rank,int n, int my_work,int *h_A,int *h_B,int *h_C,int gx_dim,int gy_dim,int bx_dim,int by_dim ) {
  int cuda_prod=0;
  int *d_A, *d_B, *d_C;
  struct timeval timecheck;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  if (my_rank == 0) {
  printf("\n**** properties: rank=%d *****\n",my_rank);
  printf("prop.name=%s\n", prop.name);
  printf("prop.multiProcessorCount=%d\n", prop.multiProcessorCount);
  printf("prop.major=%d minor=%d\n", prop.major, prop.minor);
  printf("prop.maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsDim.x=%d maxThreadsDim.y=%d maxThreadsDim.z=%d\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
  printf("prop.maxGridSize.x=%d maxGridSize.y=%d maxGridSize.z=%d\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  printf("prop.maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
  printf("prop.totalGlobalMem=%u\n", prop.totalGlobalMem);
  printf("prop.regsPerBlock=%d\n", prop.regsPerBlock);
  printf("**** properties: rank=%d *****\n",my_rank);
  printf("\n");
  }


  unsigned int my_work_size = sizeof(int) * my_work;
  unsigned int mat_size = sizeof(int) * my_work*nprocs;
  printf("rank=%d: my_work=%d data_size=%d bytes\n",my_rank,my_work,my_work_size);

  long dev_start, dev_end, dev_elapsed;
  gettimeofday(&timecheck, NULL);
  dev_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec/ 1000;

  int *h_C_on_cpu = (int *) malloc(my_work_size);

  cudaMalloc(reinterpret_cast<void **>(&d_A), my_work_size);
  cudaMalloc(reinterpret_cast<void **>(&d_B), mat_size);
  cudaMalloc(reinterpret_cast<void **>(&d_C), my_work_size);

  cudaMemcpy(d_A, h_A, my_work_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);

  by_dim = bx_dim;
  gx_dim = n/bx_dim;
  if (n%bx_dim != 0)
     gx_dim ++;
  gy_dim = n/(bx_dim*nprocs);
  if (n%(bx_dim*nprocs)!= 0)
     gy_dim ++;

  printf("bx_dim:%d by_dim:%d gx_dim:%d gy_dim:%d \n",bx_dim, by_dim,gx_dim,gy_dim);
  dim3 grid(gx_dim,gy_dim);
  dim3 threads(bx_dim,by_dim);

  mat_mult_cuda<<<grid,threads,my_work_size>>>(my_rank,n,my_work,d_A, d_B, d_C,by_dim);

  cudaMemcpy(h_C,d_C,my_work_size, cudaMemcpyDeviceToHost);

  gettimeofday(&timecheck, NULL);
  dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  dev_elapsed = dev_end - dev_start;
  
  printf("dev time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, dev_elapsed);

  fflush(stdout);
    printf("vector D with size %d\n", n*my_work);
//  output_vector(h_A,n*my_work);
  printf("vector C with size %d\n", n*my_work);
  //output_vector(h_C,n*my_work);
  matrix_multiply_cpu(my_rank,h_A,h_B,h_C_on_cpu,n,my_work);
  printf("vector C_on_cpu with size %d\n", n*my_work);
 // output_vector(h_C_on_cpu,n*my_work);
  if (compare_cpu(my_rank,h_C_on_cpu,h_C,n,my_work)) /* h_C is from dev */
    printf("\nrank=%d: Test CPU: PASS: host == dev\n", my_rank);
  else
    printf("\nrank=%d: Test CPU: FAIL: host != dev\n", my_rank);

  fflush(stdout);

  return cuda_prod;

}

void output_vector(int* data, int datasize){
     for (int i = 0; i < datasize; i++){
     	 printf("%d ",data[i]);
     }
     printf("\n");
}

void init_vec(int* data, int datasize) {
     for (int i = 0; i < datasize; i++){
     	 data[i] = rand() & 0xF;
     }
}
/*
int main(void){
    int n = 4;
    int numprocs = 2;
    int my_work = n * n / numprocs;
    int * a = new int [my_work];
    int * c = new int [my_work];
    int * d = new int [my_work];
    int * e = new int [my_work];
    init_vec(a,my_work);
    init_vec(c,my_work);
    init_vec(d,my_work);
    init_vec(e,my_work);
    int * f = new int [my_work];
    int * g = new int [my_work];
    int * h = new int [my_work];
    int * i = new int [my_work];



    int * b = new int [n*n];
    init_vec(b,n*n);
    output_vector(b,n*n);


    int tile_width = n/numprocs;
    int bx_dim = tile_width;
    int by_dim = tile_width;
    int gx_dim = n/bx_dim;
    int gy_dim = n/(bx_dim*numprocs);

    
    matrix_multiply_cuda(2,0, n,my_work, a, b, f, bx_dim, by_dim, gx_dim,gy_dim);
    matrix_multiply_cuda(2,1, n,my_work, c, b, g, bx_dim, by_dim, gx_dim,gy_dim);
//    matrix_multiply_cuda(4,3, n,my_work, d, b, h, bx_dim, by_dim, gx_dim,gy_dim);
  //  matrix_multiply_cuda(4,4, n,my_work, e, b, i, bx_dim, by_dim, gx_dim,gy_dim);
    output_vector(f,my_work);
    output_vector(g,my_work);
    output_vector(h,my_work);
    output_vector(i,my_work);
}*/
/*************************************************
  End of file
*************************************************/
