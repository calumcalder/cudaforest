#include <stdlib.h>
#include <stdio.h>
#include "forest.h"
#include "gini.h"

Forest* __assign_forest_resources(int trees, int max_depth) {
        unsigned char* node_types;
        FeatureSplit* splits;
        unsigned int max_nodes = (1 << (max_depth + 1)) - 1;
        unsigned int max_leaves = 1 << max_depth;

        cudaMalloc((void**) &node_types, trees*max_nodes*sizeof(unsigned char));
        cudaMalloc((void**) &splits, trees*max_nodes*sizeof(FeatureSplit));

        // Using intermediate to move pointers from host to device
        Forest* temp_forest = (Forest*) malloc(sizeof(Forest));
        temp_forest->node_types = node_types;
        temp_forest->splits = splits;
        temp_forest->treec = trees;
        temp_forest->max_depth = max_depth;
        temp_forest->max_nodes = max_nodes;
        temp_forest->max_leaves = max_leaves;

        Forest* device_forest;
        cudaMalloc((void**) &device_forest, sizeof(Forest));
        cudaMemcpy(device_forest, temp_forest, sizeof(Forest), cudaMemcpyHostToDevice);

        return device_forest;
}

__global__ void __assign_nodes(Forest* forest, FeatureSplit* device_splits, int depth) {
        int max_nodes = 1 << depth; // Max nodes on this level of the tree
        int tree_id = blockIdx.x; 
        int node_id = threadIdx.x + (1 << depth) - 1;

        if (threadIdx.x < max_nodes) {
                if (device_splits[tree_id].score == 0)
                        forest->node_types[tree_id*max_nodes + node_id] = FOREST_LEAF_CONSTANT;
                else {
                        forest->splits[tree_id*max_nodes + node_id] = device_splits[tree_id*max_nodes + threadIdx.x];
                        forest->node_types[tree_id*max_nodes + node_id] = FOREST_INTERNAL_CONSTANT;
                }
        }
}
/*
Forest* train_forest(DataFrame* device_data, int trees, int max_depth, int classc, int samples_per_feature, int threads_per_tree) {
        Forest* device_forest = __assign_forest_resources(trees, max_depth);
        FeatureSplit* splits = get_forest_splits_gini(device_data, classc, samples_per_feature, trees, threads_per_tree);

        DataFrame* data = (DataFrame*) malloc(sizeof(DataFrame));
        cudaMemcpy(data, device_data, sizeof(DataFrame), cudaMemcpyDeviceToHost);

        // get device_splits?
        for (int level = 0; level < max_depth; level++) {
                int max_nodes_in_level = 1 << level;
                // TODO: Get level gini splits
                // TODO: put splits in to 
                // TODO: Get next level samples
        }

        return device_forest;
}
*/
__global__ void __calculate_next_layer_node_samples(DataFrame* data, Forest* forest, int samples, bool* cur_node_samples, bool* node_samples) {
        int tree = blockIdx.x;
        int node = threadIdx.x;
        int child_node_left = (node << 1) + 1;
        int thread_id = threadIdx.y;
        int threads_per_node = blockDim.y;
        int nodes_in_cur_level = blockDim.x;
        int nodes_in_level = nodes_in_cur_level << 1;

        int child_node_left_id_in_level = child_node_left + 1 - nodes_in_level;

        // To keep in memory takes:
        // (2**depth - 1)*rows*trees bytes (as bools)
        // eg. depth 10, 1000 rows: 1mb
        //     depth 20, 1000 rows: 1gb
        // Could bring to (2**(depth - 1) + 2**(depth - 2))*rows if doing layer by layer
        // Still O(r*t*2**d) where none is dominating
        // Could reduce to node-by-node, but this seems silly.
        for (int i = thread_id; i < samples; i += threads_per_node) {
                if (cur_node_samples[tree*nodes_in_cur_level*samples + node*samples + i]) {
                        FeatureSplit feature_split = forest->splits[tree*forest->max_nodes + node];
                        bool right = feature_split.split_val < data->features[i*(data->cols - 1) + feature_split.feature];
                        // Child node right is child node left + 1, so add 1 if we want the right branch
                        node_samples[tree*nodes_in_level*samples + (child_node_left_id_in_level + 1)*samples + i] = right;
                        node_samples[tree*nodes_in_level*samples + child_node_left_id_in_level*samples + i] = !right;
                }
        }

}

bool* __get_device_node_samples(DataFrame* device_data, Forest* device_forest, bool* device_cur_node_samples, int trees, int samples, int max_nodes_in_cur_level) {
        bool* device_node_samples;
        cudaMalloc(&device_node_samples, trees*samples*(max_nodes_in_cur_level << 1)*sizeof(bool));
        cudaMemset(device_node_samples, 0, trees*samples*(max_nodes_in_cur_level << 1)*sizeof(bool)); 

        int block_count = trees;
        int threads_per_node = 1024/max_nodes_in_cur_level;
        threads_per_node = threads_per_node > 0 ? threads_per_node : 1;
        dim3 block_dims(max_nodes_in_cur_level, threads_per_node);

        __calculate_next_layer_node_samples<<<block_count, block_dims>>>(device_data, device_forest, samples, device_cur_node_samples, device_node_samples);

        cudaFree(device_cur_node_samples);

        return device_node_samples;
}


#ifdef _TEST_TRAIN_FOREST_
#include "cuda_data.h"
int main(int argc, char* argv[]) {
        DataFrame* data = readcsv("data/mnist_test_fix.csv", ',');
        DataFrame* device_data = dataframe_to_device(data);
        printf("Read data.\n");

        int samples_per_feature = atoi(argv[1]);
        int block_count = atoi(argv[2]);
        int thread_count = atoi(argv[3]);
        int max_depth = atoi(argv[4]);

        train_forest(device_data, block_count, max_depth, data->classc, samples_per_feature, thread_count);
        cudaDeviceSynchronize();
}
#endif

#ifdef _TEST_GET_NODE_SAMPLES_
#include "cuda_data.h"

Forest* forest_to_device(Forest* forest) {
        Forest* device_forest;
        cudaMalloc(&device_forest, sizeof(Forest));

        Forest* device_forest_proxy = (Forest*) malloc(sizeof(Forest));

        cudaMalloc(&(device_forest_proxy->node_types), forest->treec*forest->max_nodes*sizeof(char));
        cudaMemcpy(device_forest_proxy->node_types, forest->node_types, forest->treec*forest->max_nodes*sizeof(char), cudaMemcpyHostToDevice);
        cudaMalloc(&(device_forest_proxy->splits), forest->treec*forest->max_nodes*sizeof(FeatureSplit));
        cudaMemcpy(device_forest_proxy->splits, forest->splits, forest->treec*forest->max_nodes*sizeof(FeatureSplit), cudaMemcpyHostToDevice);
        device_forest_proxy->treec = forest->treec;
        device_forest_proxy->max_depth = forest->max_depth;
        device_forest_proxy->max_nodes = forest->max_nodes;
        device_forest_proxy->max_leaves = forest->max_leaves;

        cudaMemcpy(device_forest, device_forest_proxy, sizeof(Forest), cudaMemcpyHostToDevice);
        free(device_forest_proxy);

        return device_forest;
}

int main(int argc, char* argv[]) {
        if (argc != 2) {
                printf("Usage: ./a.out split_value\n");
                return 1;
        }
        DataFrame* data = readcsv("data/iris_train.csv", ',');
        DataFrame* device_data = dataframe_to_device(data);
        printf("Read data.\n");

        int trees = 1;
        int samples = data->rows;
        float split_val = atof(argv[1]);

        Forest* forest = (Forest*) malloc(sizeof(Forest));
        forest->splits = (FeatureSplit*) malloc(sizeof(FeatureSplit));
        forest->splits[0] = (FeatureSplit) {2, split_val, 0.1};
        forest->node_types = (unsigned char*) malloc(sizeof(char));
        forest->node_types[0] = 'i';
        forest->treec = 1;
        forest->max_depth = 2;
        forest->max_nodes = 3;
        forest->max_leaves = 2;
        printf("Made host forest\n");

        Forest* device_forest = forest_to_device(forest);
        printf("Moved forest to device\n");

        bool* device_cur_node_samples;
        cudaMalloc(&device_cur_node_samples, samples*sizeof(bool));
        cudaMemset(device_cur_node_samples, true, samples*sizeof(bool));
        printf("Allocated device memory\n");

        bool* device_new_node_samples = __get_device_node_samples(device_data, device_forest, device_cur_node_samples, trees, samples, 1);
        printf("Performed calculations\n");

        bool* new_node_samples = (bool*) malloc(2*samples*sizeof(bool));
        cudaMemcpy(new_node_samples, device_new_node_samples, 2*samples*sizeof(bool), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++) {
                printf("Sample %i: left - %s, right - %s\n",
                                i,
                                new_node_samples[i] ? "True" : "False",
                                new_node_samples[samples + i] ? "True" : "False"
                      );
        }
}
#endif
