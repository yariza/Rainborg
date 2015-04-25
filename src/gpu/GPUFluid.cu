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

int grid_X;
int grid_Y;
int grid_Z;

__device__ scalar wPoly6Kernel(Vector3s pi, Vector3s pj){
    scalar r = glm::distance(pi, pj); 
    if(r > H || r < 0)
        return 0; 

    r = ((H * H) - (r * r)); 
    r = r * r * r; // (h^2 - r^2)^3
    return r * (315.0 / (64.0 * PI * H * H * H * H * H * H * H * H * H));

}

__device__ Vector3s wSpikyKernelGrad(Vector3s pi, Vector3s pj){
    Vector3s dp = pi - pj; 
    scalar r = glm::length(dp);  
    if(r > H || r < 0)
        return Vector3s(0.0, 0.0, 0.0); 
    scalar scale = -45.0 / (PI * H * H * H * H * H * H) * (H - r) * (H - r); 
    return scale * dp; 
}

__device__ void getGridIdx(Vector3s pos, int* i, int *j, int *k){
    *i = (pos[0] - XMIN)/H;
    *j = (pos[1] - YMIN)/H;
    *k = (pos[2] - ZMIN)/H;
}

__device__ int get1DGridIdx(int i, int j, int k){
    return GRIDX * GRIDY * k + GRIDX * j + i;
}



__device__ Vector3s calcGradConstraint(Vector3s pi, Vector3s pj){
    return wSpikyKernelGrad(pi, pj)/(scalar(- P0)); 
}

__device__ Vector3s calcGradConstraintAtI(int p, Vector3s* d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd){
    Vector3s sumGrad(0.0, 0.0, 0.0);
    Vector3s pi = d_ppos[p]; 
    int gi; 
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(GRIDX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(GRIDY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(GRIDZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = get1DGridIdx(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    sumGrad += wSpikyKernelGrad(pi, d_ppos[d_grid[gi * MAX_NEIGHBORS + n]]);
                }
            }
        }
    }     
    return sumGrad / (scalar)P0; 

}

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

__global__ void applydPToPPos(Vector3s* d_ppos, Vector3s* d_dpos){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < NUM_PARTICLES){
        d_ppos[id] += d_dpos[id];
        //d_ppos[id] += Vector3s(1.0, 0, 0);
    }
}

__global__ void preserveFluidBoundaryWithUpdate(Vector3s* d_pos, Vector3s* d_ppos, Vector3s* d_dpos){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i > NUM_PARTICLES){
        return;
    }
    scalar pposX = d_ppos[i][0] + d_dpos[i][0];
    scalar pposY = d_ppos[i][1] + d_dpos[i][1];
    scalar pposZ = d_ppos[i][2] + d_dpos[i][2]; 

    if(pposX < XMIN + EPS){
        d_dpos[i][0] = XMIN + EPS - d_ppos[i][0];
        d_ppos[i][0] += d_dpos[i][0];
    }
    else if(pposX > XMAX - EPS){
        d_dpos[i][0] = XMAX - EPS - d_ppos[i][0]; 
        d_ppos[i][0] += d_dpos[i][0];
    }
    if(pposY < YMIN + EPS){
        d_dpos[i][1] = YMIN + EPS - d_ppos[i][1];
        d_ppos[i][1] += d_dpos[i][1];
    }
    else if(pposY > YMAX - EPS){
        d_dpos[i][1] = YMAX - EPS - d_ppos[i][1];
        d_ppos[i][1] += d_dpos[i][1];
    }
    if(pposZ < ZMIN + EPS){
        d_dpos[i][2] = ZMIN + EPS - d_ppos[i][2];
        d_ppos[i][2] += d_dpos[i][2];
    }
    else if(pposZ > ZMAX - EPS){
        d_dpos[i][2] = ZMAX - EPS - d_ppos[i][2];
        d_ppos[i][2] += d_dpos[i][2];
    }
}

__global__ void preserveFluidBoundary(Vector3s* d_pos, Vector3s* d_ppos, Vector3s* d_dpos){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i > NUM_PARTICLES){
        return;
    }
    scalar pposX = d_ppos[i][0] + d_dpos[i][0];
    scalar pposY = d_ppos[i][1] + d_dpos[i][1];
    scalar pposZ = d_ppos[i][2] + d_dpos[i][2]; 

    if(pposX < XMIN + EPS){
        d_dpos[i][0] = XMIN + EPS - d_ppos[i][0];
    }
    else if(pposX > XMAX - EPS){
        d_dpos[i][0] = XMAX - EPS - d_ppos[i][0]; 
    }
    if(pposY < YMIN + EPS){
        d_dpos[i][1] = YMIN + EPS - d_ppos[i][1];
    }
    else if(pposY > YMAX - EPS){
        d_dpos[i][1] = YMAX - EPS - d_ppos[i][1];
    }
    if(pposZ < ZMIN + EPS){
        d_dpos[i][2] = ZMIN + EPS - d_ppos[i][2];
    }
    else if(pposZ > ZMAX - EPS){
        d_dpos[i][2] = ZMAX - EPS - d_ppos[i][2];
    }
}

__global__ void buildGrid(Vector3s* d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id > NUM_PARTICLES){
        return;
    }
    getGridIdx(d_ppos[id], &d_gridInd[id*3], &d_gridInd[id*3+1], &d_gridInd[id*3+2]);
    int gid = get1DGridIdx(d_gridInd[id*3], d_gridInd[id*3+1], d_gridInd[id*3+2]); 

    int actgid = gid * MAX_NEIGHBORS + d_gridCount[gid];
      
    bool placed = false;
    while(!placed){
        int result = atomicCAS(&(d_grid[actgid]), -1, id);
        if(result == -1){
            placed = true;
        }
        else{
            actgid ++;
        }
    }
    //d_grid[gid * MAX_NEIGHBORS + d_gridCount[gid]] = id;
    atomicAdd(&d_gridCount[gid], 1);
}

__global__ void calcPressures(Vector3s* d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_pcalc){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p > NUM_PARTICLES){
        return;
    }
    scalar press = 0;
    int ncount = 0;
    int gi;
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(GRIDX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(GRIDY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(GRIDZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = get1DGridIdx(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    scalar pressN = wPoly6Kernel(d_ppos[p], d_ppos[d_grid[gi * MAX_NEIGHBORS + n]]); 
                    press += pressN;
                    if(pressN > 0)
                        ++ ncount; 

                }
            }
        }
    }     
    if(ncount <= MIN_NEIGHBORS && d_pcalc[p] == 0) // don't count self
        d_pcalc[p] = P0; 
    else 
        d_pcalc[p] = FP_MASS * press; // Wow I totally forgot that

}

__global__ void calcLambdas(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_pcalc, scalar *d_lambda){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p > NUM_PARTICLES)
        return;

    scalar top = -(d_pcalc[p]/P0 - 1.0);
    scalar gradSum = 0;
    scalar gradL = 0;
    Vector3s pi = d_ppos[p];

    int gi;
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(GRIDX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(GRIDY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(GRIDZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = get1DGridIdx(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    gradL = glm::length(calcGradConstraint(pi, d_ppos[d_grid[gi * MAX_NEIGHBORS + n]]));
                    gradSum += gradL * gradL;
                }
            }
        }
    }
     
    gradL = glm::length(calcGradConstraintAtI(p, d_ppos, d_grid, d_gridCount, d_gridInd));
    gradSum += gradL * gradL;
    d_lambda[p] = top / (gradSum + EPS);
} 
 
__global__ void calcdPos(Vector3s *d_ppos, Vector3s* d_dpos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_lambda){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p > NUM_PARTICLES)
        return;

    Vector3s dp(0.0, 0.0, 0.0);
    int q = 0; 
    int gi = 0;
    scalar plambda = d_lambda[p]; 
    Vector3s pi = d_ppos[p];
    scalar scorr = 0; // bla 
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(GRIDX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(GRIDY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(GRIDZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = get1DGridIdx(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    q = d_grid[gi * MAX_NEIGHBORS + n];
                    dp += (plambda + d_lambda[q] + scorr) * wSpikyKernelGrad(pi, d_ppos[q]);
                }
            }
        }
    }
    d_dpos[p] = dp / (scalar) P0;
    //d_dpos[p] = Vector3s(.1, 0, 0);
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
    grid_X = ceil(WIDTH/H);
    grid_Y = ceil(HEIGHT/H);
    grid_Z = ceil(DEPTH/H); 
    GPU_CHECKERROR(cudaMemcpyToSymbol(GRIDX, &grid_X, sizeof(int)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(GRIDY, &grid_Y, sizeof(int)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(GRIDZ, &grid_Z, sizeof(int)));


    GPU_CHECKERROR(cudaMalloc((void **)&d_grid, grid_X * grid_Y * grid_Z * MAX_NEIGHBORS * sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_gridCount, grid_X * grid_Y * grid_Z *sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_gridInd, 3 * NUM_PARTICLES * sizeof(int)));

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

void preserveOwnBoundary(bool update){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0));
    if(update)
        preserveFluidBoundaryWithUpdate<<<gridSize, BLOCKSIZE>>>(d_pos, d_ppos, d_dpos);    
    else
        preserveFluidBoundary<<<gridSize, BLOCKSIZE>>>(d_pos, d_ppos, d_dpos);

    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());



} 



void updatePredFromForce(scalar dt){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0));
    updateFromForce<<<gridSize, BLOCKSIZE>>>(d_pos, d_vel, d_ppos, dt, Vector3s(0.f, -10.0f, 0.f));    
    //updateFromForce<<<gridSize, BLOCKSIZE>>>(d_pos, d_vel, d_ppos, dt, Vector3s(0.f, 0.0f, 0.f));    
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());


}

void updateValForReals(scalar dt){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0));
    updateForReals<<<gridSize, BLOCKSIZE>>>(d_pos, d_vel, d_ppos, dt);    
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());


}

void buildGrid(){
    GPU_CHECKERROR(cudaMemset((void *)d_grid, -1, grid_X*grid_Y*grid_Z*MAX_NEIGHBORS*sizeof(int)));
    GPU_CHECKERROR(cudaMemset((void *)d_gridCount, 0, grid_X * grid_Y * grid_Z *sizeof(int)));

    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0));
    buildGrid<<<gridSize, BLOCKSIZE>>>(d_ppos, d_grid, d_gridCount, d_gridInd);
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());

     
    

}



void calculatePressures(){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0)); 
    calcPressures<<<gridSize, BLOCKSIZE>>>(d_ppos, d_grid, d_gridCount, d_gridInd, d_pcalc);  
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());

       
}

void calculateLambdas(){
    GPU_CHECKERROR(cudaMemset((void *)d_lambda, 0, NUM_PARTICLES * sizeof(scalar)));
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0)); 
    calcLambdas<<<gridSize, BLOCKSIZE>>>(d_ppos, d_grid, d_gridCount, d_gridInd, d_pcalc, d_lambda); 
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());



}

void calculatedPos(){
    GPU_CHECKERROR(cudaMemset((void *)d_dpos, 0, NUM_PARTICLES * sizeof(Vector3s)));
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0)); 
    calcdPos<<<gridSize, BLOCKSIZE>>>(d_ppos, d_dpos, d_grid, d_gridCount, d_gridInd, d_lambda); 
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());
}

void applydPToPredPos(){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0)); 
    applydPToPPos<<<gridSize, BLOCKSIZE>>>(d_ppos, d_dpos); 
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());
}

void stepSystemGPUFluid(scalar dt){
    updatePredFromForce(dt);    

    GPU_CHECKERROR(cudaMemset((void *)d_dpos, 0, NUM_PARTICLES * sizeof(Vector3s)));
    preserveOwnBoundary(true); 
    
    buildGrid(); 
    
    for(int loop = 0; loop < 1; ++loop){
        calculatePressures();
        calculateLambdas();
        calculatedPos();
        preserveOwnBoundary(false);
        applydPToPredPos();
    }
    
    updateValForReals(dt); 

}



void updateVBOGPUFluid(float *vboptr){
    int gridSize = ceil((NUM_PARTICLES * 1.0)/(BLOCKSIZE*1.0)); 
    sendToVBO<<<gridSize, BLOCKSIZE>>>(vboptr, d_pos);  
    GPU_CHECKERROR(cudaGetLastError());
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
