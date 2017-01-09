#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "data.h"

typedef struct feature_split_s {
        int feature;
        float split_val;
        float score;
} FeatureSplit;

/**
 * Function __init_rand_states
 * ---------------------------
 * Kernel to initialise random states for curand usage.
 *
 * @param rand_states The array of random states to initialise.
 *                    Should be a device pointer to blockDim.x*gridDim.x curandState_ts.
 */
__global__ void __init_rand_states(curandState_t* rand_states) {
        curand_init(0, // Seed
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
 * @param feature_split The ouput array of FeatureSplit structs to choose the best splits for each tree from.
 *
 */
__global__ void __gini_scan(DataFrame* df, unsigned int split_samples, FeatureSplit* feature_split, curandState_t* rand_states) {
        int block_thread_id = threadIdx.x;
        int grid_thread_id = blockIdx.x*blockDim.x + threadIdx.x;

        // TODO: potential speedup by mass allocation?
        size_t* counts_l = (size_t*) malloc(df->classc*sizeof(size_t));
        size_t* counts_r = (size_t*) malloc(df->classc*sizeof(size_t));
        float best_score = 1;
        feature_split[grid_thread_id].feature = -1;
        feature_split[grid_thread_id].split_val = 0;
        feature_split[grid_thread_id].score = -1;

        int thread_run_id = block_thread_id;
        while (thread_run_id < (df->cols-1)*split_samples) {
                int feature = thread_run_id/split_samples;

                memset(counts_l, 0, df->classc*sizeof(size_t));
                memset(counts_r, 0, df->classc*sizeof(size_t));
                int total_l = 0;
                int total_r = 0;

                // Get feature min and max values
                float f_min = df->features[feature];
                float f_max = df->features[feature];
                for (size_t row = 1; row < df->rows; row++) {
                        float row_value = df->features[row*(df->cols-1) + feature];
                        if (row_value < f_min)
                                f_min = row_value;
                        if (row_value > f_max)
                                f_max = row_value;
                }

                // Randomly generate split value
                float split_value = curand_uniform(&(rand_states[grid_thread_id]))*(f_max - f_min) + f_min;

                // Count class distribution for given split
                for (size_t row = 0; row < df->rows; row++) {
                        float row_value = df->features[row*(df->cols-1) + feature];
                        if (row_value < split_value) {
                                counts_l[df->classes[row]] += 1;
                                total_l += 1;
                        } else {
                                counts_r[df->classes[row]] += 1;
                                total_r += 1;
                        }
                }

                // Calculate average gini score
                float score = 0;
                for (int i = 0; i < df->classc; i++) {
                        score += (1.0f*counts_l[i]*counts_l[i])/(total_l*total_l);
                        score += (1.0f*counts_r[i]*counts_r[i])/(total_r*total_r);
                }
                score = 1 - score/2.0;

                if (score < best_score) {
                        best_score = score;
                        feature_split[grid_thread_id].feature = feature;
                        feature_split[grid_thread_id].split_val = split_value;
                        feature_split[grid_thread_id].score = score;
                }

                thread_run_id += blockDim.x; 
        }
        free(counts_l);
        free(counts_r);
}

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

        __gini_scan<<<block_count, thread_count>>>(device_data, 10, device_feature_splits, rand_states);
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

