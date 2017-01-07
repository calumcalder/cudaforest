/**
 * @author Calum Calder
 * @date 9 Nov 2016
 * 
 * Utilities for working with CSV data.
 */


#ifndef DATA_H_
#define DATA_H_
void* malloc_debug(size_t size, char* val);
/**
 * Type: DataFrame
 * ---------------
 *  Represents a number of instances of data, with each row i containing a set of features (.features[i]) and a class (.class[i]).
 *  Allows the storing of a different data type for features and the class (in this case floats and strings).
 *
 */
typedef struct dataframe_s {
        float** features;
        int* class;
        int cols;
        int classes;
        long rows;
} DataFrame;

struct train_test_split_s {
        DataFrame* train;
        DataFrame* test;
};

/**
 * Function: readcsv
 * ----------------
 *  Parses a CSV file and loads it in to memory.
 *
 *  @param filename The path to the CSV file to be read.
 *  @param delim The delimiter for values in the CSV file.
 *
 *  @return DataFrame containing features and corresponding classifications of a data set 
 *
 */
DataFrame* readcsv(const char* fname, const char delim);
float** feature_min_max(const DataFrame* data);
struct train_test_split_s train_test_split(const DataFrame* data, float ratio);
#endif
