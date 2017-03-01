#ifndef FOREST_H_
#define FOREST_H_
#include "gini.h"

#define FOREST_INTERNAL_CONSTANT 1
#define FOREST_LEAF_CONSTANT 2

typedef struct forest_s {
        unsigned char* node_types;
        FeatureSplit* splits;

        int treec;
        int max_depth;
        int max_nodes;
        int max_leaves;
} Forest;

Forest* train_forest(DataFrame* device_data, int trees, int max_depth, int classc, int samples_per_feature, int threads_per_tree);
// TODO: Take this out.
Forest* __assign_forest_resources(int trees, int max_depth);
__global__ void __calculate_next_layer_node_samples(DataFrame* data, Forest* forest, int samples, bool* cur_node_samples, bool* node_samples);

#endif 
