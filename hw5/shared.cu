#include <iostream>
using std::cerr;
using std::endl;

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#include <helper_functions.h>
#include <helper_cuda.h>

#include <unistd.h>
#include <sys/time.h>

#define MIN_TILE_WIDTH 4
#define MAX_TILE_WIDTH 16
#define THREADS_PER_BLOCK 256
#define MIN_ORDER 6
#define MAX_ORDER 10
#define MIN_N 1<<MIN_ORDER
#define MAX_N 1<<MAX_ORDER
#define MAX_PROCS 32

#define MAX_BUF_SIZE 1<<25
#define MIN_BLOCK 32
#define MAX_BLOCK 512

#define MIN_THREADS_PER_BLOCK 32
#define MAX_THREADS_PER_BLOCK 512

void init_vec(int * data, int data_size);
void output_vec(int * data, int data_size);

  // Simple 8-bit bit reversal Compute test

  #define N 256

  __global__ void bitreverse(void *data) {
     unsigned int *idata = (unsigned int*)data;
    extern __shared__ int array[];

    array[threadIdx.x] = idata[threadIdx.x];

    array[threadIdx.x] = ((0xf0f0f0f0 & array[threadIdx.x]) >> 4) |
                        ((0x0f0f0f0f & array[threadIdx.x]) << 4);
    array[threadIdx.x] = ((0xcccccccc & array[threadIdx.x]) >> 2) |
                        ((0x33333333 & array[threadIdx.x]) << 2);
    array[threadIdx.x] = ((0xaaaaaaaa & array[threadIdx.x]) >> 1) |
                         ((0x55555555 & array[threadIdx.x]) << 1);

    idata[threadIdx.x] = array[threadIdx.x];
 }

__global__ void mat_mult_cuda(int my_rank, int n, int my_work, int*d_A, int*d_B, int*d_C, int tile_width){

	   int i = blockIdx.x*blockDim.x+threadIdx.x;
	   int j = blockIdx.y*blockDim.y+threadIdx.y;
	   int index = i*tile_width+j;
	   printf("%d %d: %d\n",i,j,index);
	   extern __shared__ int a_shared[];
	   a_shared[index] = d_A[index];
	   __syncthreads();
	   
   	   extern __shared__ int b_shared[];
	   for (int k = 0; k < n*n; k++){
	   	   b_shared[my_work*n+k] = d_B[k];

	   }


	   __syncthreads();
	   int sum = 0;
	   // matrix multiply the grid not the array
	   for (int l = 0; l < my_work*n/tile_width; l++){
	       int a_index = index/n*n+l;
	       int b_index = index%n+l*n;
	       printf("%d %d %d %d\n",index,a_index,b_index,sum);
	       sum = sum+ a_shared[a_index] * b_shared[my_work*n+b_index];
	   }
	   d_C[index] = sum;


	   
}

int matrix_multiply_cuda(int nprocs, int my_rank, int n, int my_work, int *h_A, int *h_B, int *h_C, int gx_dim, int gy_dim, int bx_dim, int by_dim) {
    int* d_A, *d_B,*d_C;

    cudaMalloc(reinterpret_cast<void**>(&d_A), my_work_size);
    cudaMalloc(reinterpret_cast<void**>(&d_B), mat_size);
    cudaMalloc(reinterpret_cast<void**>(&d_C), mat_size);

    cudaMemcpy(d_A, h_A, my_work_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);

    dim3 grid(gx_dim, gy_dim);
    dim3 threads(bx_dim,by_dim);

    printf("%d %d, %d %d",gx_dim, gy_dim, bx_dim, by_dim);

    mat_mult_cuda<<<grid,threads,sizeof(int)*n*my_work>>>(my_rank,n,my_work, d_A,d_B,d_C,by_dim);

    cudaMemcpy(h_C,d_C,mat_size,cudaMemcpyDeviceToHost);

    for (int i = 0; i < n*n; i++)
    	printf("%d %d -> %d\n",h_A[i],h_B[i],h_C[i]);

    cudaFree((void*)d_A);
    cudaFree((void*)d_B);
    cudaFree((void*)d_C);
    return cuda_prod;

}
 int main(void) {
     int order = 3;
     int tile_width = 4;
     int nprocs = 2;
     int my_rank = 3;
     int n = 1<<order;
     int my_work = n/nprocs;
     int bx_dim = tile_width;
     int by_dim = bx_dim;
     int gx_dim = n/bx_dim;
     int gy_dim = n/(bx_dim*nprocs);

     int* d = new int[n*n];
     int* e = new int[n*n];
     int* f = new int[n*n];

     init_vec(d,n*my_work);
     init_vec(e,n*n);

     output_vec(d,n*my_work);
     output_vec(e,n*n);

     
     matrix_multiply_cuda(nprocs, my_rank, n, my_work, d,e,f,gx_dim, gy_dim, bx_dim,by_dim);     
     
     void *g = NULL; int i;
     unsigned int idata[N], odata[N];

     for (i = 0; i < N; i++)
         idata[i] = (unsigned int)i;

	 cudaMalloc ((void**)&g, sizeof(int)*N);
     cudaMemcpy(g, idata, sizeof(int)*N, cudaMemcpyHostToDevice);

     bitreverse<<<1, N, N*sizeof(int)>>>(g);

     cudaMemcpy(odata, g, sizeof(int)*N,
                cudaMemcpyDeviceToHost);

//     for (i = 0; i < N; i++)
  //      printf("%u -> %u\n", idata[i], odata[i]);

     cudaFree((void*)g);
     return 0;
}

void init_vec(int*data, int data_size){
     for (int i = 0; i < data_size;i++)
     	 data[i] = rand() & 0xf;
}

void output_vec(int * data, int data_size){
     for (int i = 0; i < data_size; i++)
     	 printf("%d ", data[i]);
     printf("\n");
}
