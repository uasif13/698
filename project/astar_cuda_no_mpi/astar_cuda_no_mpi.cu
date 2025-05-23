/*
	astar cuda no mpi
	compile: nvcc -ICommon astar_cuda_no_mpi.cu
	run: ./a.out < grid_size> <start> <end> <percent> <obstacle_type> <heuristic>
*/

#include <cmath>
#include <math.h>

#include <iostream>
#include <sys/time.h>
#include <random>
#include <climits>
#include <cstdint>
#include <inttypes.h>

using namespace std;
using std::cout;
using std::cerr;
using std::endl;

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

void init_graph(int* graph,int no_of_nodes, float percent);

int astar(int* graph, int start_node, int end_node, int no_of_nodes);


void output_vec(int * data, int data_size) {
  for (int i = 0; i < data_size; i ++ ) {
    printf("%d ", data[i]);
  }
  printf("\n");
}

void output_graph(int * graph, int graph_size) {
  for (int i = 0; i < graph_size*graph_size; i++) {
    if (i%graph_size == 0)
      printf("\n");
    printf("%*d", 7, graph[i]);
    
  }
  printf("\n");
}

void output_tail(int* graph, int graph_size, int tail_count) {
     uint64_t no_of_nodes = graph_size*graph_size;
     for (int i = 5 ; i >= 1; i--) {
     	 printf("%*" PRIu64  "", 3, graph[no_of_nodes-i]);
     }
     printf("\n");
}

class Graph {
public:
  int * row;
  int * col;
  int * o_index;
  int * o_array;
  uint64_t obstacle_count;
  uint64_t grid_size;
  uint64_t start;
  uint64_t end;
  Graph(int p_grid_size, float percent, int p_start, int p_end, int obstacle_type) {
    random_device rd;
    mt19937 gen(rd());
    vector<int> v_index;
    uniform_real_distribution<> dis (0.0,1/percent);
    uniform_real_distribution<> orient (0.0,1/0.5);
    grid_size = p_grid_size;
    start = p_start;
    end = p_end;
    int row, col;
    for (uint64_t i = 0; i < grid_size * grid_size; i++) {

	if (dis(gen) < 1 ? true: false) {
	  v_index.push_back(i);
	  if (orient(gen) < 1 ? true: false) {
	    for (int j = 0; j < 5; j ++) {
	      col = i %grid_size;
	      if (col+j < grid_size)
		v_index.push_back(i+j);
	    }
	  }else {
	    for (int j = 0; j < 5; j++) {
	      row = i / grid_size;
	      if (row+j < grid_size)
		v_index.push_back(i+j*grid_size);		
	    }
	  }
	}				

    }
    obstacle_count = v_index.size();
    uint64_t no_of_nodes = grid_size * grid_size;
    printf("obstacle_count: %" PRIu64 ", no_of_nodes: %" PRIu64 "\n", obstacle_count, no_of_nodes);
    o_array = new int[no_of_nodes];
    for (int i = 0; i < obstacle_count; i ++) {
      if (v_index[i] != start && v_index[i] != end)
	o_array[v_index[i]] = 1;
    }
					       
  }
    Graph(int p_grid_size, float percent, int p_start, int p_end) {
  random_device rd;
  mt19937 gen(rd());
  vector<int> v_row;
  vector<int> v_col;
  vector<int> v_index;
  int row_index, col_index;
  grid_size = p_grid_size;
  uniform_real_distribution<> dis(0.0,1/percent);
  for (int i = 0; i < grid_size*grid_size; i++) {
    
    if (i != p_start && i != p_end) {
      if (dis(gen) < 1 ? true: false) {
      row_index = i / grid_size;
      col_index = i % grid_size;
      v_row.push_back(row_index);
      v_col.push_back(col_index);
      v_index.push_back(i);
      //  printf("r:%d c:%d i:%d p_start: %d p_end: %d condition: %d\n", row_index, col_index, i, p_start, p_end, condition);
      }
    }
  }
  obstacle_count = v_row.size();
  row = new int[obstacle_count];
  col = new int[obstacle_count];
  o_index = new int[obstacle_count];
  o_array = new int[grid_size*grid_size];
  start = p_start;
  end = p_end;
  printf("\n");
  for (int i = 0; i < obstacle_count; i++) {
      row[i] = v_row[i];
      col[i] = v_col[i];
      o_index[i] = v_index[i];
      o_array[o_index[i]] = 1;
      //printf("r:%d c:%d i:%d\n", v_row[i], v_col[i], v_index[i]);
  }    
  }
};

bool checkNB(int index_f, int neighbor, int grid_size) {
  if (neighbor == -1) {
    if (index_f % grid_size == 0)
      return false;
    else
      return true;
  } else if (neighbor == 1) {
    if (index_f % grid_size == grid_size -1)
      return false;
    else
      return true;
  } else if (neighbor == -grid_size) {
    if (index_f/grid_size == 0)
      return false;
    else
      return true;
  } else {
    if (index_f/grid_size == grid_size -1)
      return false;
    else
      return true;
  }
  return false;
}

__global__ void cal_heuristic(int * d_hList, uint64_t grid_size, uint64_t end, int heuristic_type) {
	   uint64_t i = blockDim.x* blockIdx.x + threadIdx.x;
	   uint64_t row_diff, col_diff;
	   uint64_t no_of_nodes = grid_size * grid_size;
	   if (i < no_of_nodes) {
	      if (end > i){
	      	 row_diff = end -i;
		 col_diff = end -i;
		 }
	      else {
		row_diff = i-end;
		col_diff = i-end;
		}
	      row_diff /= grid_size;
	      col_diff %= grid_size;
//	      if (i == no_of_nodes - 5)
//	      	 printf("end: %" PRIu64 " i: %" PRIu64 " row_diff: %" PRIu64 " col_diff: %" PRIu64 "\n", row_diff,col_diff);
	      if (heuristic_type == 0) {
	      	 d_hList[i] = row_diff+col_diff;
	      }
	      // euclidean
	      else {
	      	 d_hList[i] = pow(row_diff*row_diff+col_diff*col_diff,0.5);
	      }
	   }
}

__global__ void astar_kernel(int * d_openList, int * d_closedList, int * d_o_array, int * d_gList, int * d_hList,int * d_neighbor, int * nVV,uint64_t grid_size) {
	       uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
	       uint64_t g_score, nb;
	       uint64_t no_of_nodes = grid_size*grid_size;
	       
	       if (i< no_of_nodes && d_openList[i] == 1) {
	       	  if (i == 100)
		     printf("\n");
	       	  *nVV =1;
		  d_openList[i] = 0;
		  d_closedList[i] = 1;
		 // printf("graph_index: %" PRIu64 " \n", i);
		  for (int j = 0; j < 4; j++) {
		      nb = i + d_neighbor[j];
		      if (d_neighbor[j] == -1 && i % grid_size != 0 || d_neighbor[j] == 1 && i % grid_size != grid_size -1 || d_neighbor[j] == -grid_size && i/grid_size == 0 || d_neighbor[j] == grid_size && i/grid_size != grid_size -1) {
		      		  
			 	       if (d_closedList[nb] != 1) {
				       	  if (d_o_array[nb] != 1) {
					  g_score = d_gList[i] + 1;
					     if (d_openList[nb] == 0) {
					     	d_openList[nb] = 1;
					     } else if (d_gList[nb] != 0 && g_score >= d_gList[nb]) {
					       continue;
					     }
					     d_gList[nb] = g_score;
					     //printf("nb_index: %d g_score: %d\n", nb, d_gList[nb]);
					  }
				       }
				       }
			}
		      }
		  }
	       


// heuristic= manhattan and euclidean
int astar(Graph * graph, string heuristic) {
  int * o_array;
  int64_t grid_size;
  uint64_t no_of_nodes;
  uint64_t start, end;
  int heuristic_type;
  if (heuristic == "manhattan")
     heuristic_type = 0;
  else
     heuristic_type = 1;

  o_array = graph -> o_array;
  grid_size = graph -> grid_size;
  no_of_nodes = grid_size*grid_size;
  start = graph -> start;
  end = graph -> end;
  int * neighbor = new int [4];

  neighbor[0] = -1;
  neighbor[1] = 1;
  neighbor[2] = -1*grid_size;
  neighbor[3] = 1*grid_size;


  int * d_hList;
  int * d_gList;
  int * d_openList;
  int * d_closedList;
  int * d_flag;
  int * d_neighbor;
  int * d_o_array;
  
  int * hList = new int[no_of_nodes];
  output_tail(hList, grid_size, 5);

  uint64_t threads = 1000;
  uint64_t grid = no_of_nodes/threads;
  if (no_of_nodes%threads != 0)
     grid ++;

  cudaMalloc(&d_hList, no_of_nodes*sizeof(int));
  cudaMalloc(&d_gList, no_of_nodes*sizeof(int));
  cudaMalloc(&d_openList, no_of_nodes*sizeof(int));
  cudaMalloc(&d_closedList, no_of_nodes*sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMalloc(&d_neighbor, 4*sizeof(int));
  cudaMalloc(&d_o_array, no_of_nodes*sizeof(int));

  cal_heuristic<<<grid,threads>>>(d_hList, grid_size, end, heuristic_type);

  cudaMemcpy(hList, d_hList, no_of_nodes*sizeof(int), cudaMemcpyDeviceToHost);

  //output_graph(hList, grid_size);
  output_tail(hList, grid_size, 5);
  
  int * openList = new int[no_of_nodes];
  int * closedList = new int[no_of_nodes];
  int * gList = new int[no_of_nodes];
  for (int i = 0; i < no_of_nodes; i++) {
      gList[i] = 0;
      openList[i] = 0;
  }

  int * newVertex = new int[1];
  *newVertex = 1;
    
  openList[start] = 1;
    
  cudaMemcpy(d_openList, openList, no_of_nodes*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neighbor, neighbor, 4*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_o_array, o_array, no_of_nodes*sizeof(int), cudaMemcpyHostToDevice);

  while (*newVertex == 1) {
    *newVertex = 0;
    
    cudaMemcpy(d_flag, newVertex, sizeof(int), cudaMemcpyHostToDevice);
    astar_kernel<<<grid,threads>>>(d_openList, d_closedList,d_o_array, d_gList, d_hList,d_neighbor, d_flag, grid_size);
    cudaMemcpy(newVertex, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    //output_vec(newVertex,1);
  }
  cudaMemcpy(gList, d_gList, no_of_nodes*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(closedList, d_closedList, no_of_nodes*sizeof(int), cudaMemcpyDeviceToHost);
  //output_graph(gList, grid_size);
  //output_graph(closedList, grid_size);
  output_tail(gList, grid_size, 5);
  if (gList[end] != 0)
     return gList[end];
  else
	return -1;
}

// single source to single destination path
int main(int argc, char* argv[]) {

  int * o_array;
  
  Graph * graph;
  int grid_size;
  int start, end;
  int obstacle_type;
  float percent;
  // manhattan, euclidean
  string heuristic;

  long astar_start, astar_end, astar_elapsed;
  
  struct timeval timecheck;

  if (argc == 7) {
    int i = 1;
    grid_size = atoi(argv[i++]);
    start = atoi(argv[i++]);
    end = atoi(argv[i++]);
    percent = atof(argv[i++]);
    obstacle_type = atoi(argv[i++]);
    heuristic = argv[i++];
  }
  else {
    grid_size = 10;
    start = 0;
    end = grid_size*grid_size - 1;
    percent = 0.05;
    obstacle_type = 1;
    heuristic = "manhattan";
  }
  if (start>= grid_size*grid_size || end>= grid_size*grid_size || start< 0 || end< 0) {
    printf("Error: start %d end %d has to be valid node[0-%d]\n",start,end, grid_size );
  }
  if (percent > 1.0 || percent <= 0.0 || (obstacle_type != 0&& obstacle_type != 1))
    printf("Error: percent %d out of bounds, obstacle type is not 0 or 1", percent);
  if (obstacle_type == 0)
    graph = new Graph(grid_size, percent, start, end);
  else
    graph = new Graph(grid_size, percent, start, end, obstacle_type);

  o_array = graph -> o_array;
  //output_graph(o_array, grid_size);
  output_tail(o_array, grid_size, 5);
  uint64_t no_of_nodes = grid_size*grid_size;
  printf("start : %d end: %d no_of_nodes : %" PRIu64 "\n",start, end, no_of_nodes);
  
  gettimeofday(&timecheck, NULL);
  astar_start = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;

  int astar_result = astar(graph,heuristic);
  printf("astar_result : %d\n", astar_result);

  gettimeofday(&timecheck, NULL);
  astar_end = (long) timecheck.tv_sec*1000 + (long)timecheck.tv_usec / 1000;

  astar_elapsed = astar_end - astar_start;

  printf("***********************\n");
  printf("no_of_nodes: %d time: %d\n", grid_size*grid_size, astar_elapsed);
    
  return 0;
}
