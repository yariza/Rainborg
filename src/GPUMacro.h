#ifdef GPU_ENABLED
#ifndef GPU_MACRO_H__
#define GPU_MACRO_H__

#include <cuda.h>

#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__))
static void gpuCheckError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }   
}


#endif 
#endif
