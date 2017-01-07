/**
 * @author Calum Calder
 * @date 9 Nov 2016
 * 
 * Utilities for working with CSV data.
 *
 * TODO: Expand for more data types.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "data.h"
#include "hashmap.h"

// TODO: Make these dynamic parameters
#define LINE_LENGTH 4096 // The max number of chars to read in per line
#define FIELD_LENGTH 30 // The max length for a field

void* malloc_debug(size_t size, char* val) {
        void* ptr = malloc(size);
//printf("%s: %p - %p\n", val, ptr, ptr + size);
        return ptr;
}

/**
 * Function: __csv_fieldc
 * ----------------------
 *  Counts the number of fields in the CSV at @param filename, delimeted by @param delim.
 *
 *  @param filename The path to the CSV file to be read.
 *  @param delim The delimiter for values in the CSV file.
 *
 *  NOTE: Not robust to incorrectly formatted CSVs (eg. 'jagged' CSVs).
 *  TODO: Fix failed parse for escaped \ preceding the delimiter.
 *
 */
int __csv_fieldc(const char* filename, const char delim) {
        int fields = 0;
        FILE* stream;
        char line[LINE_LENGTH];

        stream = fopen(filename, "r");
        if (stream == NULL)
                return -1;

        fgets(line, LINE_LENGTH, stream);
        for (int i = 0; i < strlen(line); i++)
                if (line[i] == delim && line[i-1] != '\\')
                        fields++;

        fclose(stream);

        return fields + 1;
}

/**
 * Function: __csv_linec
 * ---------------------
 *  Counts the lines in the file at @param filename.
 *
 *  @param filename The path to the file to be loaded.
 *
 *  @return A long containing the number of lines in the file.
 *
 *  TODO: Do the maths on whether a long is really required here.
 */
size_t __csv_linec(const char* filename) {
        long lines = 0;
        FILE* stream;
        char line[LINE_LENGTH];
        stream = fopen(filename, "r");
        
        if (stream == NULL) {
                perror("Could not open file for reading");
                return -1;
        }


        while (fgets(line, LINE_LENGTH, stream))
                lines++;
        
        fclose(stream);
        return lines;
}

/**
 * Function: __csv_readline
 * ------------------------
 *  Reads a single line from a CSV file stream in to memory.
 *
 *  @param stream The file stream to read the next line from.
 *  @param delim The delimiter separating values in the CSV file.
 *  @param fields The number of fields in the CSV file.
 *
 *  @return An pointer to @param fields many strings.
 */
char** __csv_readline(FILE* stream, char delim, int fields) {
        char** result; 
        char line[LINE_LENGTH];
        char current_string[FIELD_LENGTH];
        int current_string_index = 0;
        int current_field_index = 0;

        // Set up results array
        result = (char**) malloc(fields*sizeof(char*));
        //printf("result: %p\n", result);
        for (int i = 0; i < fields; i++)
                result[i] = (char*) malloc(FIELD_LENGTH*sizeof(char));

        // Read in the line
        if (fgets(line, LINE_LENGTH, stream)) {
                for (int i = 0; i < strlen(line) - 1; i++) {
                        if (line[i] == '\\' && line[i-1] != '\\') {
                                continue;
                        } else if (line[i] == delim && line[i-1] != '\\') {
                                // Terminate and copy current string to results
                                current_string[current_string_index] = '\0';
                                strcpy(result[current_field_index], current_string);

                                // reset current string and indices
                                memset(current_string, 0, sizeof(current_string));
                                current_string_index = 0;
                                current_field_index++;
                        } else {
                                current_string[current_string_index] = line[i];
                                current_string_index++;
                        }
                }
                // Terminate and copy current string to results
                current_string[current_string_index] = '\0';
                strcpy(result[current_field_index], current_string);
        } else {
                return NULL;
        }
        return result;
}


/**
 * Function: __test__file
 * ----------------------
 *  Tests that a given file loads using this CSV parser.
 *
 *  @param filename The path of the file to be parsed.
 *  @param delim The delimiter of variables in the CSV file.
 *
 */
void __test_file(const char* filename, const char delim) {
        int fieldc = __csv_fieldc(filename, delim);
        long linec = __csv_linec(filename);

        DataFrame* df = readcsv(filename, delim);
                        
        printf("%ld lines loaded from file %s.\n", linec, filename);
        for (int i = 0; i < 10 && i < df->rows; i++) {
                printf("%i: ", i);
                for (int j = 0; j < fieldc - 1; j++)
                        printf("%f, ", df->features[i*(df->cols-1) + j]);
                printf("%i\n", df->classes[i]);
        }

                
}


/**
 * Function: __rawtodataframe
 * ------------------------------
 *  Converts raw data read in as a 2D-Array-like of strings in to a dataframe.
 *
 *  @param rawdata The raw data to be converted.
 *
 *  @return DataFrame filled from raw data.
 */
DataFrame* __rawtodataframe(const char*** rawdata, int cols, long rows) {
        DataFrame* df = (DataFrame*) malloc(sizeof(DataFrame));
        df->cols = cols;
        df->rows = rows;

        //printf("Allocating memory for DF\n");
        df->features = (float*) malloc((cols-1)*rows*sizeof(float));
        df->classes = (int*) malloc(rows*sizeof(int));

       // printf("Converting from string data to floats\n");

        int class_idx = 0;
        HashMap* class_map = make_hashmap(16);
        for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols - 1; j++)
                        df->features[i*(df->cols-1) + j] = atof(rawdata[i][j]);
                if (get_hashmap(class_map, rawdata[i][cols - 1]) == -1) {
                        put_hashmap(class_map, rawdata[i][cols - 1], class_idx);
                        class_idx++;
                }
                df->classes[i] = get_hashmap(class_map, rawdata[i][cols-1]);
        }
        df->classc = class_idx;
        return df;
}

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
 *  TODO: Fix failed parsing of complex escape patterns.
 */
 DataFrame* readcsv(const char* filename, const char delim) {
        // printf("Getting field count\n");
        size_t fieldc = __csv_fieldc(filename, delim);
        // printf("Getting line count\n");
        size_t linec = __csv_linec(filename);
        char*** data;

        // Initialise data to empty strings
        //printf("Allocating data\n");
        data = (char***) malloc(linec*sizeof(char**));
        for (int i = 0; i < linec; i++) {
                //data[i] = (char**) malloc(fieldc*sizeof(char*));
                //for (int j = 0; j < fieldc; j++)
                        //data[i][j] = (char*) malloc(FIELD_LENGTH*sizeof(char), "data[i][j]");
        }

        FILE* stream = fopen(filename, "r");
        //printf("Opening file for reading\n");
        char** line;
        int i = 0;
        for (int i = 0; i < linec; i++) {
                data[i] = __csv_readline(stream, delim, fieldc);
        }

        fclose(stream);
        //printf("Closed file\n");
        DataFrame* df = __rawtodataframe((const char***) data, fieldc, linec);

        //printf("Freeing data\n");
        for (int i = 0; i < linec; i++) {
                for (int j = 0; j < fieldc; j++) {
                        free(data[i][j]);
                }
                free(data[i]);
        }
        free(data);
        //printf("Freed raw data\n");

        return df;
}

//float** feature_min_max(const DataFrame* data) {
//        float** ret;
//        int features = data->cols - 1;
//        ret = (float**) malloc(features*sizeof(float*));
//        for (int i = 0; i < features; i++)
//                ret[i] = (float*) malloc(2*sizeof(float));

//        for (int i = 0; i < data->rows; i++) {
//                for (int j = 0; j < data->cols - 1; j++) {
//                        if (ret[j][1] < data->features[i][j])
//                                ret[j][1] = data->features[i][j];
//                        if (ret[j][0] > data->features[i][j])
//                                ret[j][0] = data->features[i][j];
//                }
//        }

//        printf("1\n");
//        return ret;
//}

float** feature_min_max(const DataFrame* data) {
        size_t features = data->cols - 1;
        float** ret = (float**) malloc(features*sizeof(float*));
        for (int i = 0; i < features; i++) {
                ret[i] = (float*) malloc(2*sizeof(float));
                ret[i][0] = 1000000000;
                ret[i][1] = -1000000000;
        }

        for (int i = 0; i < data->rows; i++)
        for (int j = 0; j < features; j++) {
                float fval = data->features[i*(data->cols-1) + j];
                if (ret[j][0] > fval)
                        ret[j][0] = fval;
                if (ret[j][1] < fval)
                        ret[j][1] = fval;
        }

        return ret;
}

struct train_test_split_s train_test_split(const DataFrame* data, float ratio) {
        DataFrame* train;
        DataFrame* test;
        char* row_chosen = (char*) malloc(data->rows*sizeof(char));
        int rows_train = (int) (ratio*data->rows);
        int rows_test = data->rows - rows_train;

        train->features = (float*) malloc(data->cols*rows_train*sizeof(float*));
        train->classes = (int*) malloc(rows_train*sizeof(int));

        test->features = (float*) malloc(data->cols*rows_test*sizeof(float*));
        test->classes = (int*) malloc(rows_test*sizeof(int));

        struct train_test_split_s ret = { train, test };
        return ret;
}

#ifdef _TEST_DATA_C_
int main(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
                __test_file(argv[i], ',');
                if (i < argc - 1)
                        printf("\n");
        }

        return 0;
}
#endif
