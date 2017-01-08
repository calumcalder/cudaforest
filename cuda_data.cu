#include <stdio.h>
#include "cuda_data.h"
#include "data.h"

DataFrame* dataframe_to_device(const DataFrame* df) {
        /*
         Transferring via an intermediate to allow copying of 
         device pointers while preserving old dataframe pointers
         */
        DataFrame* temp_df = (DataFrame*) malloc(sizeof(DataFrame));
        temp_df->rows = df->rows;
        temp_df->cols = df->cols;
        temp_df->classc = df->classc;

        DataFrame* ddf;

        cudaMalloc((void**) &ddf, sizeof(DataFrame));
        cudaMalloc((void**) &(temp_df->features), df->rows*(df->cols-1)*sizeof(float));
        cudaMalloc((void**) &(temp_df->classes), df->rows*sizeof(int));

        cudaMemcpy(temp_df->features, df->features, (df->cols-1)*df->rows*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_df->classes, df->classes, df->rows*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ddf, temp_df, sizeof(DataFrame), cudaMemcpyHostToDevice);

        return ddf;
}

DataFrame* dataframe_to_host(const DataFrame* ddf) {
        DataFrame* df = (DataFrame*) malloc(sizeof(DataFrame));
        cudaMemcpy(df, ddf, sizeof(DataFrame), cudaMemcpyDeviceToHost);

        float* device_features = df->features;
        int* device_classes = df->classes;

        df->features = (float*) malloc((df->cols-1)*df->rows*sizeof(float));
        df->classes = (int*) malloc(df->rows*sizeof(int));

        cudaMemcpy(df->features, device_features,
                   (df->cols-1)*df->rows*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(df->classes, device_classes,
                   df->rows*sizeof(int), cudaMemcpyDeviceToHost);

        return df;
}

#ifdef _TEST_CUDA_DATA_CU_
int main(int argc, char* argv[]) {
        DataFrame* orig_df = readcsv("data/iris_train.csv", ',');
        DataFrame* device_df = dataframe_to_device(orig_df);
        DataFrame* new_df = dataframe_to_host(device_df);

        printf("Orig prt: %p, device ptr: %p, new prt: %p\n", orig_df, device_df, new_df);
        printf("orig_df: rows %li cols %i classes %i\n", orig_df->rows, orig_df->cols, orig_df->classc);
        printf("new_df: rows %li cols %i classes %i\n", new_df->rows, new_df->cols, new_df->classc);
        for (int i = 0; i < 10 && i < new_df->rows; i++) {
                printf("%i: ", i);
                for (int j = 0; j < new_df->cols - 1; j++)
                        printf("%f, ", new_df->features[i*(new_df->cols-1) + j]);
                printf("%i\n", new_df->classes[i]);
        }

}

#endif
