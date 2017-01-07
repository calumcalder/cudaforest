#ifndef CUDA_DATA_H
#define CUDA_DATA_H
#include "data.h"

DataFrame* dataframe_to_device(const DataFrame* df);
DataFrame* dataframe_to_host(const DataFrame* ddf);

#endif
