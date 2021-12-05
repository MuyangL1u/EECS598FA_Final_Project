#include "libwb/wb.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <map>
#include <fstream>
#include <utility>
#include <assert.h>
#include "benchmark.h"
#include <curand.h>
#include <cuda.h>


typedef NodeWeight<int64_t, float> WNode;
typedef EdgePair<NodeID, WNode> Edge;
typedef pvector<Edge> EdgeList;

static std::random_device rd;
static std::mt19937 rng(rd());

// Print datasets for debugging?
// CAUTION: This will print the entire training/testing datasets
//          Can fill up the terminal and slow down the program!
bool print_datasets = false;

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		assert(err == cudaSuccess); \
	} \
}

double RandomNumberGenerator() 
{
    static std::uniform_real_distribution<double> uid(0,1);//  
    return uid(rng);
}


// void randomGenerator(double *dataHost, int number, unsigned long long seed)
// {   
//     double *dataDev;
//     cudaMalloc( (void **) &dataDev, number * sizeof(double) );
 
//     curandGenerator_t gen;
//     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//    	curandSetPseudoRandomGeneratorSeed(gen, seed);
//     curandGenerateUniformDouble(gen, dataDev, number);
//     curandDestroyGenerator(gen);
 
//     cudaMemcpy(dataHost, dataDev, number * sizeof(double), cudaMemcpyDeviceToHost);
//     cudaFree(dataDev);
 
//     return;
// }

#define BLOCK_SIZE 64
#define TILE_WIDTH 16

void __global__ device_random_walk_kernel(
	int m_walk_length, int n_walks_per_node, 
	int total_num_nodes, double *rnumber, 
	int64_t * d_p_scan_list, int64_t * d_v_list, 
	float * d_w_list, int64_t *d_global_walk, 
	int64_t *d_outdegree);

void WriteWalkToAFile(
	NodeID* global_walk, 
	int num_nodes, 
	int max_walk,
	int num_walks_per_node,
	std::string walk_filename);

void compute_random_walk_call( 
	WGraph &g, 
    int max_walk_length,
    int num_walks_per_node,
	std::string walk_filename
){
	max_walk_length++;
	int64_t *host_p_sum;
	int64_t *device_p_sum;

	int64_t *host_value;
	int64_t *device_value;

	float *host_weight;
	float *device_weight;

//extracted from g.outdegree() to use in the cuda kernel
	int64_t *host_outdegree;
	int64_t *device_outdegree;

	//random number list
	double* host_r_list;
	double * device_r_list;

	NodeID *global_walk = new NodeID[g.num_nodes() * max_walk_length * num_walks_per_node];
	NodeID *device_walk;

  	host_p_sum = new int64_t[g.num_nodes() + 1];
	host_value = new int64_t[g.num_edges()];
	host_weight = new float[g.num_edges()];
	host_outdegree = new int64_t[g.num_nodes()];
	host_r_list = new double[num_walks_per_node*(max_walk_length-1)* g.num_nodes()];
	host_p_sum[0] = 0;

	for(NodeID i = 0; i < g.num_nodes(); ++i) {
		host_p_sum[i + 1] = host_p_sum[i] + g.out_degree(i);
		host_outdegree[i] = g.out_degree(i);
		int cnt = 0;
		for(auto v: g.out_neigh(i)){
		  int64_t idx = host_p_sum[i] + cnt;
		  host_value[idx] = v.v;
		  host_weight[idx] = v.w;
		  cnt++;
		}
	}

	cudaCheck(cudaMalloc((void**) &device_p_sum, (g.num_nodes() + 1) * sizeof(int64_t)));
	cudaCheck(cudaMalloc((void**) &device_value, g.num_edges() * sizeof(int64_t)));
	cudaCheck(cudaMalloc((void**) &device_weight, g.num_edges() * sizeof(float)));
	cudaCheck(cudaMalloc((void**) &device_outdegree, g.num_nodes() * sizeof(int64_t)));
	cudaCheck(cudaMalloc((void**) &device_walk, g.num_nodes() * max_walk_length * num_walks_per_node * sizeof(int64_t)));
  
	cudaCheck(cudaMemcpy(device_p_sum, host_p_sum , (g.num_nodes() + 1) * sizeof(int64_t) , cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(device_value, host_value , g.num_edges() * sizeof(int64_t) , cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(device_weight, host_weight , g.num_edges() * sizeof(float) , cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(device_outdegree, host_outdegree, g.num_nodes() * sizeof(int64_t), cudaMemcpyHostToDevice));
	
	dim3 dimGrid = ceil((float)g.num_nodes()/BLOCK_SIZE);
  	dim3 dimBlock = BLOCK_SIZE;

	for (int i=0; i < (max_walk_length-1)*num_walks_per_node* g.num_nodes(); i++ ){
	    host_r_list[i] =  RandomNumberGenerator();
	}
	
	cudaCheck(cudaMalloc((void **)&device_r_list,sizeof(double)*(max_walk_length-1)*num_walks_per_node* g.num_nodes()));
	cudaCheck(cudaMemcpy(device_r_list,host_r_list,sizeof(double)*num_walks_per_node*(max_walk_length-1)* g.num_nodes(),cudaMemcpyHostToDevice));
	
	std::cout << "Computing random walk for " << g.num_nodes() << " nodes and " 
      	<< g.num_edges() << " edges." << std::endl;
	Timer t;
	t.Start();

	for(int i = 0; i < num_walks_per_node/10; i++){
		for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
			// random_number = nerator();
    		device_random_walk_kernel<<<dimGrid,dimBlock>>>(max_walk_length,num_walks_per_node,g.num_nodes(),device_r_list, 
    		device_p_sum, device_value, device_weight, device_walk, device_outdegree);
			cudaDeviceSynchronize();
		}
	}

	cudaCheck(cudaMemcpy(global_walk, device_walk, sizeof(int64_t)*g.num_nodes() * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost));
	t.Stop();

	PrintStep("[TimingStat] Random walk time (s):", t.Seconds());
 	WriteWalkToAFile(global_walk, g.num_nodes(), 
    max_walk_length, num_walks_per_node, walk_filename);
  	
  	cudaFree(device_p_sum);
  	cudaFree(device_value);
 	cudaFree(device_weight);
	cudaFree(device_outdegree);
  	cudaFree(device_walk);
	cudaFree(device_r_list);

	delete[] host_r_list;
	delete[] host_p_sum;
	delete[] host_value;
	delete[] host_weight; 
	delete[] host_outdegree;
	delete[] global_walk;
}

// __device__ double curand_uniform_double (curandState_t *state);

void __global__ device_random_walk_kernel(
  int m_walk_length, int n_walks_per_node, 
  int total_num_nodes,	double *rnumber, 
  int64_t * d_p_scan_list, int64_t * d_v_list, 
  float * d_w_list, int64_t *d_global_walk, 
  int64_t *d_outdegree){

    int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	// int64_t tid = threadIdx.x;
	// __shared__ int64_t cache[TILE_WIDTH];
	// if(i < TILE_WIDTH){
	// 	cache[i] = 0;
	// }
	// __syncthreads();
		if(i >= total_num_nodes){
			return;
		}

		long long int w;
	    for(int w_n = 0; w_n < n_walks_per_node; ++w_n) {
			// d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + 0] = i;
			d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + 0] = i;
			float prev_time_stamp = 0;
			int64_t src_node = i;
			int walk_cnt;
			for(walk_cnt = 1; walk_cnt < m_walk_length; ++walk_cnt) {
			  int valid_neighbor_cnt = 0;
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){
				if(d_w_list[w] > prev_time_stamp){
				  valid_neighbor_cnt++;
				  break;
				}
			  }
			  if(valid_neighbor_cnt == 0) {
				break;
			  }
			  float min_bound = d_w_list[d_p_scan_list[src_node]];
			  float max_bound = d_w_list[d_p_scan_list[src_node]];
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){
				if(d_w_list[w] < min_bound)
				  min_bound = d_w_list[w];
				if(d_w_list[w] > max_bound)
				  max_bound = d_w_list[w];
			  }
			  float time_boundary_diff = (max_bound - min_bound);

			  if(time_boundary_diff < 0.0000001){
				for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){ // We randomly pick 1 neighbor, we just pick the first	
					if(d_w_list[w] > prev_time_stamp){
						// d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
						d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
						src_node = d_v_list[w];
						prev_time_stamp = d_w_list[w];
						break;
					}
				}
				continue; 
			  }
			  
			double exp_summ = 0;            
			for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){		
				if(d_w_list[w] > prev_time_stamp){
				  exp_summ += exp((float)(d_w_list[w]-prev_time_stamp)/time_boundary_diff);
				}
			}

			// __shared__ double exp_summs[BLOCK_SIZE];
			// long long int distance = d_p_scan_list[src_node+1] - d_p_scan_list[src_node];
			// int remain = distance % BLOCK_SIZE;
			// if(distance > BLOCK_SIZE){
			// 	if (remain != 0){
			// 		for(w=0; w < remain; w++){
			// 			if(d_w_list[d_p_scan_list[src_node+1] - w] > prev_time_stamp){
			// 				exp_summs[0] += exp((float)(d_w_list[d_p_scan_list[src_node+1] - w]-prev_time_stamp)/time_boundary_diff);
			// 			}
			// 		}
			// 		__syncthreads();
			// 	}
			// 	for(w = 0; w < distance / BLOCK_SIZE; w++){		
			// 		if(d_w_list[d_p_scan_list[src_node] + w] > prev_time_stamp){
			// 		  exp_summs[tid] += exp((float)(d_w_list[d_p_scan_list[src_node] + w*tid]-prev_time_stamp)/time_boundary_diff);
			// 		}
			// 	}
			// 	__syncthreads();
			// }
			// else {
			// 	for(w=0; w < remain; w++){
			// 		if(d_w_list[d_p_scan_list[src_node+1] - w] > prev_time_stamp){
			// 			exp_summs[0] += exp((float)(d_w_list[d_p_scan_list[src_node+1] - w]-prev_time_stamp)/time_boundary_diff);
			// 		}
			// 	}
			// 	__syncthreads();
			// }
			
			// exp_summ = exp_summs[0];
			// for(int m = 0; m< total_num_nodes/TILE_WIDTH; m++){
			// 	int idx = m * TILE_WIDTH + i;
			// 	if(i < TILE_WIDTH){
			// 		cache[i] = d_w_list[idx];
			// 	}
			// 	__syncthreads();
			// 	if( idx >= d_p_scan_list[src_node] && idx < d_p_scan_list[src_node+1]){
			// 		for(int n=0; n < TILE_WIDTH; n++){
			// 			if(cache[n] > prev_time_stamp){
			// 				exp_summ += exp((float)(cache[n]-prev_time_stamp)/time_boundary_diff);
			// 			}
			// 		}	
			// 		__syncthreads();
			// 	}
			// }
				
			
			  double curCDF = 0, nextCDF = 0;
			//   double random_number = rnumber;
			  double random_number = rnumber[( w_n * (m_walk_length-1) * n_walks_per_node ) + (i * (m_walk_length-1) ) + walk_cnt];
			bool fall_through = false;
			  for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){		
				if(d_w_list[w] > prev_time_stamp){
					nextCDF += (exp((float)(d_w_list[w]-prev_time_stamp)/time_boundary_diff) * 1.0 / exp_summ);
					if(nextCDF >= random_number && curCDF <= random_number) {
					//   d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
					  d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
					  src_node = d_v_list[w];
					  prev_time_stamp = d_w_list[w];
					  fall_through = true;
					  break;
				  } else {
					  curCDF = nextCDF;
				  }
				}
			  }
			  if(!fall_through){
				for(w = d_p_scan_list[src_node]; w < d_p_scan_list[src_node+1]; w++){ // This line should not be reached anyway (reaching this line means something is wrong). But just for testing, we randomly pick 1 neighbor, we just pick the first  
				  if(d_w_list[w] > prev_time_stamp){
					// d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = d_v_list[w];
					d_global_walk[( i * m_walk_length * n_walks_per_node ) + ( w_n * m_walk_length ) + walk_cnt] = d_v_list[w];
					src_node = d_v_list[w];
					prev_time_stamp = d_w_list[w];
					break; 
				  }
				}
			  }
			}
			if (walk_cnt != m_walk_length){	
			// d_global_walk[( total_num_nodes * w_n * m_walk_length) + ( i * m_walk_length ) + walk_cnt] = -1;
				d_global_walk[( i * n_walks_per_node * m_walk_length) + ( w_n * m_walk_length ) + walk_cnt] = -1;
			}
			
		}
}

void WriteWalkToAFile(
	NodeID* global_walk, 
	int num_nodes, 
	int max_walk,
	int num_walks_per_node,
	std::string walk_filename) 
  {
	std::ofstream random_walk_file(walk_filename);
	for(int w_n = 0; w_n < num_walks_per_node; ++w_n) {
	  for(NodeID iter = 0; iter < num_nodes; iter++) {
		NodeID *local_walk = 
		  global_walk + 
		  ( iter * max_walk * num_walks_per_node ) +
		  ( w_n * max_walk );
		for (int i = 0; i < max_walk; i++) {
			if (local_walk[i] == -1)
			  break;
			random_walk_file << local_walk[i] << " ";
		}
		random_walk_file << "\n";
	  }
	}
	random_walk_file.close();
}
  

int main(int argc, char **argv) {
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
  
	  compute_random_walk_call(
		/* temporal graph */ g, 
		/* max random walk length */ max_walk_length,
		/* number of rwalks/node */ num_walks_per_node,
		/* filename of random walk */ "out_random_walk.txt"
	  );
	}
	return 0;
}
  