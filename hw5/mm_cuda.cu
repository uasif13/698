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
//	printf("%d %d\n", j, k);
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

	// a_width x a_width
	int n_sq = a_width * a_width;
/*	tile_width = by_dim * gy_dim
	____
	|  |	tile_height = bx_dim*gx_dim
	____
	
*/
	int num_tiles_per_row = a_width/tile_width;
	if (a_width%tile_width != 0) {
	   num_tiles_per_row++;
	}
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	printf("%d\n",blockIdx.x* blockDim.x);

	// tiled row
	int r = my_rank/num_tiles_per_row;
	// tiled col
	int c = my_rank%num_tiles_per_row;
	int tile_height = a_width/(blockDim.x*blockIdx.x);
	// tiled row*tile_height+tiled_col*tile_width = top left of tile
	
	if (i == 0 && j == 0) {
	   for (int m = 0; m < tile_width; m++) {
	       for (int n = 0; n < tile_height; n++) {
	       	int shared_index = r*tile_height+m*a_width+c*tile_width+n;
		int a_index = m*tile_width+n;
		int b_index = shared_index;
		printf("shared pointers: %d %d %d %d %d %d %d\n",i,j,r,c, shared_index, a_index, b_index);
		a_shared[shared_index] = d_a[a_index];
		b_shared[shared_index] = d_b[b_index];

	       }
	   }
	}
	/*



	*/
	
	__syncthreads();

	if (i == 0 && j == 0 && my_rank == 0) {
		for (int a = 0; a < 16; a++) {
			printf("a:%d avalue: %d\n",a, a_shared[a]);
			printf("b:%d bvalue: %d\n",a, b_shared[a]);
		}

		}
		    int sum = 0;
   	
	
		    int k = 0, l = 0;
		    printf("matmulti row and col %d %d %d\n", r, c,i*tile_width+j);
		    while (k < r+a_width &&l< a_width*a_width){
		    	  int as_index = r*tile_height+i*a_width+k;
			  int bs_index = l+c*tile_width+j;
			  printf("inner row, col, index, k,l,sum %d,%d,%d %d %d %d %d %d\n",r, c, tile_width*i+j, k,l,sum,a_shared[as_index],b_shared[bs_index]);
			  sum += a_shared[as_index]*b_shared[bs_index];
			  l = l +a_width;
			  k ++;

			  }
			   d_c[i*tile_width+j] = sum;
			  printf("%d ",d_c[i*tile_width+j]);
	
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

  unsigned int my_work_size = sizeof(int) * my_work * n;
  unsigned int mat_size = sizeof(int) * n * n;
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
  gy_dim = n/(bx_dim*nprocs);

  printf("bx_dim:%d by_dim:%d gx_dim:%d gy_dim:%d \n",bx_dim, by_dim,gx_dim,gy_dim);
  dim3 grid(gx_dim,gy_dim);
  dim3 threads(bx_dim,by_dim);

  mat_mult_cuda<<<grid,threads,n*n>>>(my_rank,n,my_work,d_A, d_B, d_C,by_dim);

  cudaMemcpy(h_C,d_C,my_work_size, cudaMemcpyDeviceToHost);

  gettimeofday(&timecheck, NULL);
  dev_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  dev_elapsed = dev_end - dev_start;
  
  printf("dev time: rank=%d: %d procs: %ld msecs\n",
	   my_rank, nprocs, dev_elapsed);

  fflush(stdout);
    printf("vector D with size %d\n", n*my_work);
  output_vector(h_A,n*my_work);
  printf("vector C with size %d\n", n*my_work);
  output_vector(h_C,n*my_work);
  matrix_multiply_cpu(my_rank,h_A,h_B,h_C_on_cpu,n,my_work);
  printf("vector C_on_cpu with size %d\n", n*my_work);
  output_vector(h_C_on_cpu,n*my_work);
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
