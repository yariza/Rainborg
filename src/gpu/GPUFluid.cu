#ifdef GPU_ENABLED
#include "GPUFluid.h"

#define BLOCKSIZE 256

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

__global__ void sendToVBO(float *vbo, Vector3s* d_pos){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < NUM_PARTICLES){
        vbo[id*4+0] = d_pos[id][0];
        vbo[id*4+1] = d_pos[id][1];
        vbo[id*4+2] = d_pos[id][2];
        vbo[id*4+3] = 1.0f;

    }

}



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

    GPU_CHECKERROR(cudaMemset((void *)d_vel, 0, NUM_PARTICLES * sizeof(Vector3s)));

    /*
    curandState *state;
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_pcalc, NUM_PARTICLES); // temporarily store here I guess
    curandDestroyGenerator(gen);
    */

    Vector3s *h_pos;
    GPU_CHECKERROR(cudaMallocHost((void **)&h_pos, NUM_PARTICLES * sizeof(Vector3s)));
    float x; 
    float y; 
    float z;
    for(int i = 0; i < NUM_PARTICLES; ++i){
         x = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
         y = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
         z = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
         h_pos[i] = Vector3s(x, y, z);
    }
    GPU_CHECKERROR(cudaMemcpy((void *)d_pos, (void *)h_pos, NUM_PARTICLES * sizeof(Vector3s), cudaMemcpyHostToDevice));

    GPU_CHECKERROR(cudaFreeHost(h_pos));


}

//void 


void stepSystemGPUFluid(){
    // 
    // predict positions
    // init dpos to 0
    // preserve boundary
    // apply dp to predpos
    

}

void updateVBOGPUFluid(float *vboptr){

    


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
