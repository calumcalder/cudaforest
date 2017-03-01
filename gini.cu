#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "gini.h"
#include "forest.h"
#include "data.h"
#include "cuda_data.h"

/**
 * Function __init_rand_states
 * ---------------------------
 * Kernel to initialise random states for curand usage.
 *
 * @param rand_states The array of random states to initialise.
 *                    Should be a device pointer to blockDim.x*gridDim.x curandState_ts.
 */
__global__ void __init_rand_states(curandState_t* rand_states) {
        curand_init(clock64() + threadIdx.x, // Seed
                    blockIdx.x*blockDim.x + threadIdx.x, // Unique sequence id
                    0,
                    &(rand_states[blockIdx.x*blockDim.x + threadIdx.x]));
}

/**
 * Function: __gini_scan
 * ---------------------
 * Kernel for basis of Gini scoring for feature and split value selection.
 *
 * @param df The DataFrame representing the dataset.
 * @param split_samples The number of sample split values per feature to attempt a split on.
 * @param rand_states A pointer to blockDim.x*gridDim.x curandStates_ts.
 * @param feature_split The ouput array of FeatureSplit structs to choose the best splits for each tree from.
 *
 */
__global__ void __gini_scan(DataFrame* df, bool* node_samples, int node, int max_nodes_in_level, unsigned int split_samples, curandState_t* rand_states, FeatureSplit* feature_split) {
        int block_thread_id = threadIdx.x;
        int tree = blockIdx.x;
        int grid_thread_id = blockIdx.x*blockDim.x + threadIdx.x;
        
        if (block_thread_id > df->rows*split_samples)
                return;

        // TODO: potential speedup by mass allocation?
        size_t* counts_l = (size_t*) malloc(df->classc*sizeof(size_t));
        size_t* counts_r = (size_t*) malloc(df->classc*sizeof(size_t));
        float best_score = 2.0;
        feature_split[grid_thread_id].feature = -1;
        feature_split[grid_thread_id].split_val = 0;
        feature_split[grid_thread_id].score = -1;

        //int thread_run_id = block_thread_id;
        //while (thread_run_id < (df->cols-1)*split_samples) {
        for (int thread_run_id = block_thread_id; thread_run_id < (df->cols - 1)*split_samples; thread_run_id += blockDim.x) {
                int feature = thread_run_id/split_samples;

                memset(counts_l, 0, df->classc*sizeof(size_t));
                memset(counts_r, 0, df->classc*sizeof(size_t));
                int total_l = 0;
                int total_r = 0;

                // Get feature min and max values
                float f_min = df->features[feature];
                float f_max = df->features[feature];
                for (size_t row = 1; row < df->rows; row++) {
                        if (node_samples[tree*max_nodes_in_level*df->rows + node*df->rows + row]) {
                                float row_value = df->features[row*(df->cols-1) + feature];
                                if (row_value < f_min)
                                        f_min = row_value;
                                if (row_value > f_max)
                                        f_max = row_value;
                        }
                }

                // Randomly generate split value
                float split_value = curand_uniform(&(rand_states[grid_thread_id]))*(f_max - f_min) + f_min;

                // Count class distribution for given split
                for (size_t row = 0; row < df->rows; row++) {
                        float row_value = df->features[row*(df->cols-1) + feature];
                        if (node_samples[tree*max_nodes_in_level*df->rows + node*df->rows + row]) {
                                if (row_value < split_value) {
                                        counts_l[df->classes[row]] += 1;
                                        total_l += 1;
                                } else {
                                        counts_r[df->classes[row]] += 1;
                                        total_r += 1;
                                }
                        }
                }

                // Calculate average gini score
                float score = 0;
                for (int i = 0; i < df->classc; i++) {
                        score += (1.0f*counts_l[i]*counts_l[i])/(total_l*total_l);
                        score += (1.0f*counts_r[i]*counts_r[i])/(total_r*total_r);
                }
                score = 1 - score/2.0;

                if (score <= best_score) {
                        best_score = score;
                        feature_split[grid_thread_id].feature = feature;
                        feature_split[grid_thread_id].split_val = split_value;
                        feature_split[grid_thread_id].score = score;
                }

                //thread_run_id += blockDim.x; 
        }

        free(counts_l);
        free(counts_r);
}

/**
 * Function __consolidate_feature_splits 
 * -------------------------------------
 * Kernel to consolidate feature splits using a binary tree recursion method.
 *
 * @param feature_splits A pointer to the feature splits to consolidate.
 *        Should be a FeatureSplit[trees*threads_per_tree].
 * @param consolidate_feature_splits A pointer to the output consolidated splits.
 *        Shape should be FeatureSplit[tree_count].
 *
 */
__global__ void __consolidate_feature_splits_into_forest(FeatureSplit* feature_splits, Forest* forest, int node_tree_id) {
        int thread_id = threadIdx.x;
        int tree_id = blockIdx.x;
        int threads_per_tree = blockDim.x;

        int modulo;
        for (modulo = 2; 1.0*threads_per_tree/modulo > 1.0; modulo = modulo << 1) {
                if (thread_id % modulo == 0 && forest->treec*threads_per_tree > tree_id*threads_per_tree + thread_id) {
                        FeatureSplit left_split = feature_splits[tree_id*threads_per_tree + thread_id];
                        FeatureSplit right_split = feature_splits[tree_id*threads_per_tree + thread_id + (modulo >> 1)];
                        if ((left_split.score > right_split.score || left_split.score < 0) && right_split.score > 0) {
                                feature_splits[tree_id*threads_per_tree + thread_id] = right_split;
                        }
                }
                __syncthreads();
        }

        if (thread_id == 0) {
                forest->splits[tree_id*forest->max_nodes + node_tree_id] = feature_splits[tree_id*threads_per_tree];
        }

}

void set_forest_layer_splits_gini(DataFrame* device_data, Forest* device_forest, bool* device_node_samples, int classc, int samples_per_feature, int tree_count, int threads_per_tree, int max_nodes_in_layer) {
        curandState_t* rand_states;
        FeatureSplit* device_feature_splits;
        int block_count = tree_count;
        int thread_count = threads_per_tree;

        cudaMalloc((void**) &rand_states, block_count*thread_count*sizeof(curandState_t));
        cudaMalloc((void**) &device_feature_splits, block_count*thread_count*sizeof(FeatureSplit));

        __init_rand_states<<<block_count, thread_count>>>(rand_states);
        for (int node_layer_id = 0; node_layer_id < max_nodes_in_layer; node_layer_id++) {
                int node_tree_id = node_layer_id + max_nodes_in_layer - 1;
                cudaMemset((void*) device_feature_splits, 0, block_count*thread_count*sizeof(FeatureSplit));

                __gini_scan<<<block_count, thread_count>>>(device_data, device_node_samples, node_layer_id, max_nodes_in_layer, samples_per_feature, rand_states, device_feature_splits);
                __consolidate_feature_splits_into_forest<<<block_count, thread_count>>>(device_feature_splits, device_forest, node_tree_id);
        }

        cudaFree(device_feature_splits);
        cudaFree(rand_states);
}
/*
FeatureSplit* get_forest_splits_gini(DataFrame* device_data, int classc, int samples_per_feature, int tree_count, int threads_per_tree) {
        //unsigned long long int* device_counts;
        curandState_t* rand_states;
        FeatureSplit* device_feature_splits;
        FeatureSplit* device_consolidated_feature_splits;
        int block_count = tree_count;
        int thread_count = threads_per_tree;

     // cudaMalloc((void**) &device_counts, classc*sizeof(unsigned long long int));
     // cudaMemset((void*) device_counts, 0, classc*sizeof(unsigned long long int));

        cudaMalloc((void**) &rand_states, block_count*thread_count*sizeof(curandState_t));
        __init_rand_states<<<block_count, thread_count>>>(rand_states);

        cudaMalloc((void**) &device_feature_splits, block_count*thread_count*sizeof(FeatureSplit));
        cudaMemset((void*) device_feature_splits, 0, block_count*thread_count*sizeof(FeatureSplit));

        __gini_scan<<<block_count, thread_count>>>(device_data, samples_per_feature, rand_states, device_feature_splits);

        cudaMalloc((void**) &device_consolidated_feature_splits, block_count*sizeof(FeatureSplit));
        cudaMemset((void*) device_consolidated_feature_splits, 0, block_count*thread_count*sizeof(FeatureSplit));
        __consolidate_feature_splits<<<block_count, thread_count>>>(device_feature_splits, device_consolidated_feature_splits);

        cudaFree(device_feature_splits);
        cudaFree(rand_states);

        // Consolidate results
        // TODO: move to device for speedup?
    //  FeatureSplit* feature_splits = (FeatureSplit*) malloc(block_count*thread_count*sizeof(FeatureSplit));
    //  cudaError e = cudaMemcpy(feature_splits, device_feature_splits, 
    //                  block_count*thread_count*sizeof(FeatureSplit), cudaMemcpyDeviceToHost);
    //  printf("%i\n", e);

    //  FeatureSplit* best_splits = (FeatureSplit*) malloc(block_count*sizeof(FeatureSplit));
    //  for (int block = 0; block < block_count; block++) {
    //          FeatureSplit best_split = feature_splits[block*thread_count];
    //          for (int thread = 1; thread < thread_count; thread++)
    //                  if (feature_splits[block*thread_count + thread].score < best_split.score 
    //                      && feature_splits[block*thread_count + thread].score >= 0)
    //                          best_split = feature_splits[block*thread_count + thread];
    //          best_splits[block] = best_split;
    //  }


        return device_consolidated_feature_splits;
}
*/
#ifdef _TEST_GINI_SCAN_
#include "cuda_data.h"
int main(int argc, char* argv[]) {
        DataFrame* data = readcsv("data/mnist_test_fix.csv", ',');
        DataFrame* device_data = dataframe_to_device(data);
        printf("Read data.\n");

        unsigned long long int* device_counts;
        cudaMalloc((void**) &device_counts, data->classc*sizeof(unsigned long long int));
        cudaMemset((void*) device_counts, 0, data->classc*sizeof(unsigned long long int));
        printf("Allocated device memory.\n");

        int block_count = atoi(argv[1]);
        int thread_count = atoi(argv[2]);

        curandState_t* rand_states;
        cudaMalloc((void**) &rand_states, block_count*thread_count*sizeof(curandState_t));
        __init_rand_states<<<block_count, thread_count>>>(rand_states);
        cudaThreadSynchronize();

        FeatureSplit* device_feature_splits;
        cudaMalloc((void**) &device_feature_splits, block_count*thread_count*sizeof(FeatureSplit));
        cudaMemset((void*) device_feature_splits, 0, block_count*thread_count*sizeof(FeatureSplit));
        printf("Allocated device split memory.\n");

        __gini_scan<<<block_count, thread_count>>>(device_data, 10, rand_states, device_feature_splits);
        printf("Completed scan.\n");

        FeatureSplit* feature_splits = (FeatureSplit*) malloc(block_count*thread_count*sizeof(FeatureSplit));
        cudaError e = cudaMemcpy(feature_splits, device_feature_splits, 
                        block_count*thread_count*sizeof(FeatureSplit), cudaMemcpyDeviceToHost);
        printf("%i\n", e);

        for (int i = 0; i < block_count*thread_count; i++) {
                printf("ID: %i, feature: %i, split value: %f, score %f\n",
                              i,
                              feature_splits[i].feature,
                              feature_splits[i].split_val,
                              feature_splits[i].score);
        }

        return 0;
}
#endif
#ifdef _TEST_FOREST_SPLITS_GINI_
#include "cuda_data.h"
int main(int argc, char* argv[]) {
        if (argc != 4) {
                printf("Usage: ./a.out samples_per_feature tree_count thread_count\n");
                return 1;
        }
        DataFrame* data = readcsv("data/mnist_test_fix.csv", ',');
        DataFrame* device_data = dataframe_to_device(data);
        printf("Read data.\n");

        int samples_per_feature = atoi(argv[1]);
        int block_count = atoi(argv[2]);
        int thread_count = atoi(argv[3]);

        FeatureSplit* splits = get_forest_splits_gini(device_data, data->classc, samples_per_feature, block_count, thread_count);
        FeatureSplit* host_splits = (FeatureSplit*) malloc(block_count*sizeof(FeatureSplit));
        cudaMemcpy(host_splits, splits, block_count*sizeof(FeatureSplit), cudaMemcpyDeviceToHost);

        for (int i = 0; i < block_count; i++) {
                printf("ID: %i, feature: %i, split value: %f, score %f\n",
                              i,
                              host_splits[i].feature,
                              host_splits[i].split_val,
                              host_splits[i].score
                      );
        }

        return 0;
}
#endif
#ifdef _TEST_SET_FOREST_LAYER_SPLITS_GINI_
int main(int argc, char* argv[]) {
        int tree_count = 2;
        int max_depth = 1;

        Forest* device_forest = __assign_forest_resources(tree_count, max_depth);
        DataFrame* data = readcsv("data/iris.csv", ',');
        DataFrame* device_data = dataframe_to_device(data);
        printf("Read data.\n");
        int samples = data->rows;

        int max_nodes_per_layer = 1;
        int max_nodes_per_next_layer = max_nodes_per_layer*2;

        bool* device_cur_node_samples;
        cudaMalloc(&device_cur_node_samples, tree_count*max_nodes_per_layer*samples*sizeof(bool));
        cudaMemset(device_cur_node_samples, true, tree_count*max_nodes_per_layer*samples*sizeof(bool));

        int classc = data->classc;
        int samples_per_feature = 128;
        int threads_per_tree = 512;
        threads_per_tree = threads_per_tree > samples ? samples : threads_per_tree;
        set_forest_layer_splits_gini(device_data, device_forest, device_cur_node_samples, classc, samples_per_feature, tree_count, threads_per_tree, max_nodes_per_layer);

        bool* device_next_node_samples;
        cudaMalloc(&device_next_node_samples, tree_count*max_nodes_per_next_layer*samples*sizeof(bool));
        cudaMemset(device_next_node_samples, false, tree_count*max_nodes_per_next_layer*samples*sizeof(bool));

        int block_count = tree_count;
        int threads_per_node = 1024/max_nodes_per_layer;
        threads_per_node = threads_per_node > 0 ? threads_per_node : 1;
        dim3 block_dims(max_nodes_per_layer, threads_per_node);
        __calculate_next_layer_node_samples<<<block_count, block_dims>>>(device_data, device_forest, data->rows, device_cur_node_samples, device_next_node_samples);

        cudaFree(device_cur_node_samples);
        device_cur_node_samples = device_next_node_samples;
        max_nodes_per_layer = max_nodes_per_next_layer;
        max_nodes_per_next_layer = max_nodes_per_layer*2;
        set_forest_layer_splits_gini(device_data, device_forest, device_cur_node_samples, classc, samples_per_feature, tree_count, threads_per_tree, max_nodes_per_layer);

        Forest* proxy_forest = (Forest*) malloc(sizeof(Forest));
        cudaMemcpy(proxy_forest, device_forest, sizeof(Forest), cudaMemcpyDeviceToHost);

        FeatureSplit* splits = (FeatureSplit*) malloc(proxy_forest->treec*proxy_forest->max_nodes*sizeof(FeatureSplit));
        cudaMemcpy(splits, proxy_forest->splits, proxy_forest->treec*proxy_forest->max_nodes*sizeof(FeatureSplit), cudaMemcpyDeviceToHost);
        printf("%i\n", proxy_forest->treec*proxy_forest->max_nodes);

        printf("Feature: %i, Split val: %f, Gini Impurity Score: %f\n",
                        splits[0].feature,
                        splits[0].split_val,
                        splits[0].score
              );
        printf("Feature: %i, Split val: %f, Gini Impurity Score: %f\n",
                        splits[1].feature,
                        splits[1].split_val,
                        splits[1].score
              );
        printf("Feature: %i, Split val: %f, Gini Impurity Score: %f\n\n",
                        splits[2].feature,
                        splits[2].split_val,
                        splits[2].score
              );
        printf("Feature: %i, Split val: %f, Gini Impurity Score: %f\n",
                        splits[3].feature,
                        splits[3].split_val,
                        splits[3].score
              );
        printf("Feature: %i, Split val: %f, Gini Impurity Score: %f\n",
                        splits[4].feature,
                        splits[4].split_val,
                        splits[4].score
              );
        printf("Feature: %i, Split val: %f, Gini Impurity Score: %f\n",
                        splits[5].feature,
                        splits[5].split_val,
                        splits[5].score
              );

        return 0;
}
#endif
