#ifndef __GPU_HELPER_H__
#define __GPU_HELPER_H__
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__))
static void gpuCheckError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    }
}

#endif
