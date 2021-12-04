#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <map>
#include <fstream>

#include "benchmark.h"

typedef NodeWeight<NodeID, WeightT> WNode;
typedef EdgePair<NodeID, WNode> Edge;
typedef pvector<Edge> EdgeList;
typedef std::vector<std::pair<NodeID, WeightT>> TempNodeVector;
typedef std::pair<NodeID, WeightT> TNode;
typedef std::vector<double> DoubleVector;

static std::random_device rd;
static std::mt19937 rng(rd());

// Print datasets for debugging?
// CAUTION: This will print the entire training/testing datasets
//          Can fill up the terminal and slow down the program!
bool print_datasets = false;

#include "rwalk.h"

int main(int argc, char* argv[]) {

  CLApp cli(argc, argv, "link-prediction");

  if (!cli.ParseArgs())
    return -1;

  // Data structures
  WeightedBuilder b(cli);
  EdgeList el;
  WGraph g = b.MakeGraph(&el);

  // Read parameter configuration file
  cli.read_params_file();

  // Parameter initialization
  int   max_walk_length     =   cli.get_max_walk_length();
  int   num_walks_per_node  =   cli.get_num_walks_per_node();

  // Compute temporal random walk
  for(int i=0; i<20; ++i) {
    compute_random_walk(
      /* temporal graph */ g, 
      /* max random walk length */ max_walk_length,
      /* number of rwalks/node */ num_walks_per_node,
      /* filename of random walk */ "out_random_walk.txt"
    );
  }

  return 0;
}