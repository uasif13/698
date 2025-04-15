#include <iostream>
#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
#include <limits.h>

using namespace std;

using std::cout;
using std::cerr;
using std::endl;

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define NO_OF_NODES 9;
class CSR {
   public:
      int* srcPtrs;
      int srcPtrs_size;
      int* dst;
      int dst_size;
      CSR(int * graph, int no_of_nodes){
      	  vector<int> v_srcPtrs;
	  vector<int> v_dst;
	  v_srcPtrs.push_back(0);
	  for (int i = 0; i < no_of_nodes*no_of_nodes; i ++) {
//	      printf("index: %d graph: %d srcPtrs_size: %d dst_size %d\n", i, graph[i], v_srcPtrs.size(), v_dst.size());
      	      int row = i / no_of_nodes;
	      int col = i % no_of_nodes;
	      if (graph[i] != 0){
		 // new row
		 if (row == v_srcPtrs.size()) {
		    v_srcPtrs.push_back(v_dst.size());
		 }
		 v_dst.push_back(col);
	      }
	      if (col == no_of_nodes - 1 && row == v_srcPtrs.size())
	      	 v_srcPtrs.push_back(v_dst.size());
	  }
	  v_srcPtrs.push_back(v_dst.size());
	  srcPtrs_size = v_srcPtrs.size();
	  srcPtrs = new int[srcPtrs_size];
	  dst_size = v_dst.size();
	  dst = new int[dst_size];
	  for (int i = 0; i < srcPtrs_size; i++){
	      srcPtrs[i] = v_srcPtrs[i];
	  }
	  for (int i = 0; i < dst_size; i ++){
	      dst[i] = v_dst[i];
	  }
      }
      CSR (int* n_srcPtrs, int* n_dst){
      	  srcPtrs = n_srcPtrs;
	  dst = n_dst;
      }
};

class CSC {
      public:
	int * dstPtrs;
	int* src;
	int dstPtrs_size;
	int src_size;
        CSC(int* graph, int no_of_nodes) {
		 vector<int> v_dstPtrs;
		 vector<int> v_src;
		 for (int i = 0; i < no_of_nodes*no_of_nodes; i++) {
		     int col = i / no_of_nodes;
		     int row = i% no_of_nodes;
		     if (graph[row*no_of_nodes+col] != 0) {
		     	if (col == v_dstPtrs.size()){
			   v_dstPtrs.push_back(v_src.size());
			}
			v_src.push_back(row);
		     }
		     if (row == no_of_nodes -1 && col == v_dstPtrs.size()){
		     	v_dstPtrs.push_back(v_src.size());
		     }
		 }
		 v_dstPtrs.push_back(v_src.size());
		 dstPtrs_size = v_dstPtrs.size();
		 src_size = v_src.size();
		 dstPtrs = new int[dstPtrs_size];
		 src = new int[src_size];
		 for (int i = 0; i < dstPtrs_size; i ++){
		     dstPtrs[i] = v_dstPtrs[i];
		 }
		 for (int i = 0; i < src_size; i++){
		     src[i] = v_src[i];
		 }
	}
};

class COO {
      public:
	int * srcPtrs;
	int * dstPtrs;
	int size;
	COO (int * graph, int no_of_nodes) {
	    vector<int> v_srcPtrs;
	    vector<int> v_dstPtrs;
	    for (int i = 0; i < no_of_nodes*no_of_nodes; i++) {
	    	if (graph[i] != 0) {
		   v_srcPtrs.push_back(i/no_of_nodes);
		   v_dstPtrs.push_back(i%no_of_nodes);
		}
	    }
	    size = v_srcPtrs.size();
	    srcPtrs = new int[size];
	    dstPtrs = new int[size];
	    for (int i = 0; i < size; i++) {
	    	srcPtrs[i] = v_srcPtrs[i];
		dstPtrs[i] = v_dstPtrs[i];
	    }
	}
};

__global__ void bfs_kernel(int * d_srcPtrs, int* d_dstPtrs, int size_of_edges, int no_of_nodes, int start_node, int end_node, int * level, int currLevel, int * newVertexVisited){
	int i = blockDim.x*blockIdx.x+threadIdx.x;
	printf("edge: %d, currLevel: %d\n", i, currLevel);
	if (i < size_of_edges) {
	   int src_vertex = d_srcPtrs[i];
	   if (level[src_vertex] == currLevel -1){
	      printf("vertex: %d, currLevel: %d, level: %d, edge:[%d-%d]\n", src_vertex, currLevel, level[src_vertex], d_srcPtrs[i], d_dstPtrs[i]);

	      	      int neighbor = d_dstPtrs[i];
		      printf("n: %d n_lvl: %d\n", neighbor, level[neighbor]);
		      if (level[neighbor] == 100) {
		      	 level[neighbor] = currLevel;
			 *newVertexVisited = 1;
//		      	 printf("n: %d n_lvl: %d\n", neighbor, level[neighbor]);			 
	      }
	   }
	}
}

int bfs(COO* coo, int no_of_nodes, int start_node, int end_node, int grid, int threads){
    int* d_srcPtrs;
    int * d_dstPtrs;
    int * d_level;
    int* srcPtrs = coo -> srcPtrs;
    int * dstPtrs = coo -> dstPtrs;
    int size = coo -> size;

    int * level = new int[no_of_nodes];
    printf("create level array\n");
    level[0] = 0;
    printf("set first index\n");
    for (int i = 1; i < no_of_nodes; i ++){
    	     level[i] = 100;
    }
    printf("set other indices\n");
    int * newVertex = new int[1];
    *newVertex = 1;
    int currLevel = 1;
    printf("initialize kernel arrays\n");
    cudaMalloc(&d_srcPtrs, size*sizeof(int));
    cudaMalloc(&d_dstPtrs, size*sizeof(int));
    cudaMalloc(&d_level, no_of_nodes*sizeof(int));
    printf("initialize level array\n");
        printf("copy arrays\n");
    cudaMemcpy(d_srcPtrs, srcPtrs, size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstPtrs, dstPtrs, size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level, no_of_nodes*sizeof(int),cudaMemcpyHostToDevice);    
    printf("enter kernel loop\n");
    while (*newVertex == 1){
    	  *newVertex = 0;

	  bfs_kernel<<<grid, threads>>>(d_srcPtrs, d_dstPtrs,size, no_of_nodes, start_node, end_node, d_level, currLevel, newVertex);
	  currLevel++;
    }
    cudaMemcpy(level, d_level, no_of_nodes*sizeof(int), cudaMemcpyDeviceToHost);
    if (level[end_node] != 100)
       return 1;
    else
	return -1;

    
}
void init_graph(int * graph, int no_of_nodes){
    int book_graph[] = { 0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};

    for (int i = 0; i < no_of_nodes*no_of_nodes; i++){
    	// graph[i] = rand() & (no_of_nodes-1);
	graph[i] = book_graph[i];
    }
}

void output_vec(int * arr, int arr_size) {
     for (int i = 0; i < arr_size; i++){
     	 printf("%d ", arr[i]);
     }
     printf("\n");
}

int main(int argc, char* argv[]){
    
    int no_of_nodes = NO_OF_NODES;
    int start_node = 0;
    int end_node = no_of_nodes -1;

    if (argc > 1){
       int i = 1;
       no_of_nodes = atoi(argv[i++]);
       start_node = atoi(argv[i++]);
       end_node = atoi(argv[i++]);
    }
    int * graph = new int[no_of_nodes*no_of_nodes];
    printf("create graph\n");
    init_graph(graph, no_of_nodes);
    printf("create csr \n");
    CSR * csr = new CSR(graph, no_of_nodes);
    CSC * csc = new CSC(graph, no_of_nodes);
    COO * coo = new COO(graph, no_of_nodes);
    int * o_coo_srcPtrs = coo -> srcPtrs;
    int * o_coo_dstPtrs = coo -> dstPtrs;
    int o_size = coo -> size;
    output_vec(o_coo_srcPtrs,o_size);
    output_vec(o_coo_dstPtrs, o_size);
    int * o_dstPtrs = csc -> dstPtrs;
    int * o_src = csc -> src;
    int o_src_size =csc -> src_size;
    int o_dstPtrs_size = csc -> dstPtrs_size;
    output_vec(o_dstPtrs,o_dstPtrs_size);
    output_vec(o_src,o_src_size);
    int * o_srcPtrs = csr -> srcPtrs;
    int o_srcPtrs_size = csr -> srcPtrs_size;
    int * o_dst = csr -> dst;
    int o_dst_size = csr -> dst_size;
    output_vec(o_srcPtrs, o_srcPtrs_size);
    output_vec(o_dst, o_dst_size);
    int grid = 1;
    int threads = o_size;
    int bfs_result = bfs(coo, no_of_nodes,start_node, end_node, grid, threads);
    printf("bfs_result: %d\n", bfs_result);
}