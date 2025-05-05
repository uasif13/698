#include <igraph.h>
#include <stdio.h>

int main(int argc, char * argv) {
  igraph_integer_t num_vertices = 50;
  igraph_integer_t num_edges = 75;
  igraph_real_t diameter;
  igraph_t graph;
  FILE * output;

  output = fopen("output.dot", "w");
  igraph_rng_seed(igraph_rng_default(), 42);

  igraph_erdos_renyi_game_gnm(
			      &graph, num_vertices, num_edges,
			      IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS
			      );

  if (output != NULL) {
    igraph_write_graph_dot(&graph, output);
  }
  igraph_diameter(
		  &graph, &diameter,
		  /* from = */ NULL, /* to == */ NULL,
		  /* vertex_path = */ NULL, /*edge_path = */ NULL,
		  IGRAPH_UNDIRECTED, /* unconn= */ true
		  );
  printf("Diameter of a random graph with average degree %g: %g \n",
	 2.0 * igraph_ecount(&graph) / igraph_vcount(&graph),
	 (double) diameter);

  igraph_destroy(&graph);

  return 0;
}
