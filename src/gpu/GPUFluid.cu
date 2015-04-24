#ifdef GPU_ENABLED
#include "GPUFluid.h"

#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__))
static void gpuCheckError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }   
}


__constant__ int GRIDX; 
__constant__ int GRIDY;
__constant__ int GRIDZ;

scalar *d_pos;
scalar *d_vel; 
scalar *d_ppos;
scalar *d_dpos;
scalar *d_pcalc; 
scalar *d_lambda; 

int *d_grid;
int *d_gridCount;
int *d_gridInd; 


void initGPUFluid(){
    // allocate memory on GPU
    // Initialize positions, velocities
    std::cout << "Initializing things" << std::endl;

    GPU_CHECKERROR(cudaMalloc((void **)&d_pos, NUM_PARTICLES * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_vel, NUM_PARTICLES * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_ppos, NUM_PARTICLES * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_dpos, NUM_PARTICLES * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_pcalc, NUM_PARTICLES * sizeof(scalar)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_lambda, NUM_PARTICLES * sizeof(scalar)));
    int grid_X = ceil(WIDTH/H);
    int grid_Y = ceil(HEIGHT/H);
    int grid_Z = ceil(DEPTH/H); 
    GPU_CHECKERROR(cudaMemcpyToSymbol(GRIDX, &grid_X, sizeof(int)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(GRIDY, &grid_Y, sizeof(int)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(GRIDZ, &grid_Z, sizeof(int)));


    GPU_CHECKERROR(cudaMalloc((void **)&d_grid, grid_X * grid_Y * grid_Z * MAX_NEIGHBORS * sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_gridCount, grid_X * grid_Y * grid_Z *sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_gridInd, NUM_PARTICLES * sizeof(int)));

}

void 


void stepSystemGPUFluid(){
    // accumulate Forces
    // predict positions
    // init dpos to 0
    // preserve boundary
    // apply dp to predpos

}

void cleanUpGPUFluid(){

    GPU_CHECKERROR(cudaFree(d_pos));
    GPU_CHECKERROR(cudaFree(d_vel));
    GPU_CHECKERROR(cudaFree(d_ppos));
    GPU_CHECKERROR(cudaFree(d_dpos));
    GPU_CHECKERROR(cudaFree(d_pcalc));
    GPU_CHECKERROR(cudaFree(d_lambda));

    GPU_CHECKERROR(cudaFree(d_grid));
    GPU_CHECKERROR(cudaFree(d_gridCount));
    GPU_CHECKERROR(cudaFree(d_gridInd));






}

#endif
