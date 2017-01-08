#include <stdlib.h>
#include <stdio.h>
#include "data.h"

typedef struct class_split_s {
        int class_id;
        float split_val;
} ClassSplit;

/**
 * Function: __gini_prescan
 * ------------------------
 * Kernel to calculate the distribution of classes throughout the dataset.
 *
 * @param df The DataFrame object representing the data.
 * @param counts The output class counts, in uncosolidated form with counts per thread.
 *               Should be a block of df->classc*gridsize unsigned long longs.
 *
 */
__global__ void __gini_prescan(DataFrame* df, unsigned long long int* counts) {
        size_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;
        size_t grid_size = gridDim.x*blockDim.x; // Assuming 1D grid and 1D blocks

        size_t row = thread_id;

        while (row < df->rows) {
                int class_id = df->classes[row];
                atomicAdd(&(counts[class_id]), 1);
                row += grid_size;
        }
}

//__global__ void __gini(DataFrame* df, float*) {
//        int feature = threadIdx.x;
//}

//ClassSplit select_class_gini(DataFrame* df) {
//        ClassSplit class_split;
//        return class_split;
//}

#ifdef _TEST_GINI_PRESCAN_
#include "cuda_data.h"
int main(int argc, char* argv[]) {
        DataFrame* data = readcsv("data/iris_test.csv", ',');
        DataFrame* device_data = dataframe_to_device(data);
        printf("Read data.\n");

        unsigned long long int* device_counts;
        cudaMalloc((void**) &device_counts, data->classc*sizeof(unsigned long long int));
        cudaMemset((void*) device_counts, 0, data->classc*sizeof(unsigned long long int));
        printf("Allocated device memory.\n");

        int block_count = 2;
        int thread_count = 16;
        __gini_prescan<<<block_count, thread_count>>>(device_data, device_counts);
        printf("Completed prescan.\n");

        unsigned long long int* counts =
                (unsigned long long int*) malloc(data->classc*sizeof(unsigned long long int));
        cudaMemcpy(counts, device_counts,
                        data->classc*sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < data->classc; i++)
                printf("Class %i: %llu\n", i, counts[i]);

        return 0;
}
#endif
