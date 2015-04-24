#ifdef GPU_ENABLED
#include "GPUFluid.h"

#define BLOCKSIZE 256

#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__))
static void gpuCheckError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        //exit(EXIT_FAILURE);
    }   
}


__constant__ int GRIDX; 
__constant__ int GRIDY;
__constant__ int GRIDZ;

Vector3s *d_pos;
Vector3s *d_vel; 
Vector3s *d_ppos;
Vector3s *d_dpos;
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
        //vbo[id*4+0] = 1.0f; 
        //vbo[id*4+1] = 1.0f;
        //vbo[id*4+2] = 1.0f;
        //vbo[id*4+3] = 1.0f;

    }

}

__global__ void updateFromForce(Vector3s* d_pos, Vector3s* d_vel, Vector3s* d_ppos, scalar dt, Vector3s force){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < NUM_PARTICLES){
        d_vel[id] += force * dt;
        d_ppos[id] = d_pos[id] + d_vel[id]*dt; 
    }
}

__global__ void updateForReals(Vector3s* d_pos, Vector3s* d_vel, Vector3s* d_ppos, scalar dt){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < NUM_PARTICLES){
        d_vel[id] = (d_ppos[id] - d_pos[id])/dt;
        d_pos[id] = d_ppos[id];
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

void preserveOwnBoundary(){


} 



void updatePredFromForce(scalar dt){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0));
    updateFromForce<<<gridSize, BLOCKSIZE>>>(d_pos, d_vel, d_ppos, dt, Vector3s(0.f, -10.0f, 0.f));    
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());


}

void updateValForReals(scalar dt){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0));
    updateForReals<<<gridSize, BLOCKSIZE>>>(d_pos, d_vel, d_ppos, dt);    
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());


}

void stepSystemGPUFluid(scalar dt){
    updatePredFromForce(dt);    
    //preserveOwnBoundary(); 
    //applydPToPredPos();
    //buildGrid(); 
    updateValForReals(dt); 

}



void updateVBOGPUFluid(float *vboptr){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0)); 
    sendToVBO<<<gridSize, BLOCKSIZE>>>(vboptr, d_pos);  
    //GPU_CHECKERROR(cudaGetLastError());
    // Is sad the first call, then fine
    GPU_CHECKERROR(cudaThreadSynchronize());

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
