#ifndef GINI_H_
#define GINI_H_
#include "data.h"

// TODO: Consider moving to a more general file
typedef struct feature_split_s {
        int feature;
        float split_val;
        float score;
} FeatureSplit;

FeatureSplit* get_forest_splits_gini(DataFrame* device_data, int classc, int samples_per_feature, int tree_count, int threads_per_tree);

#endif
