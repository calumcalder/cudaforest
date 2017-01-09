#ifndef CUDA_DATA_H_
#define CUDA_DATA_H_
#include "data.h"

DataFrame* dataframe_to_device(const DataFrame* df);
DataFrame* dataframe_to_host(const DataFrame* ddf);

#endif
