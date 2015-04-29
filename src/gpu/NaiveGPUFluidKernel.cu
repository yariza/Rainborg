#ifdef GPU_ENABLED
#include "NaiveGPUFluidKernel.h"
#include "GPUHelper.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>

const int naive_BLOCKSIZE_1D = 512;

bool naive_deviceHappy = true;

__constant__ scalar c_h;
__constant__ scalar c_minX;
__constant__ scalar c_maxX;
__constant__ scalar c_minY;
__constant__ scalar c_maxY;
__constant__ scalar c_minZ;
__constant__ scalar c_maxZ;
__constant__ scalar c_eps;    
__constant__ int c_gridSizeX;
__constant__ int c_gridSizeY;
__constant__ int c_gridSizeZ;


// helper functions for getting grid indices
__device__ void naive_getGridLocation(Vector3s pos, int &i, int &j, int &k);
__device__ int naive_getGridIndex(int i, int j, int k);
__device__ void naive_getGridLocationFromIndex(int id, int &i, int &j, int &k);

// helper function for getting grid size
__host__ void naive_getGridSize(FluidBoundingBox* fbox, scalar h,
                               int &gridSizeX, int &gridSizeY, int &gridSizeZ);

__device__ Vector3s naive_getFluidVolumePosition(FluidVolume& volume, int k);


 // GPU functions

__device__ __host__ scalar naive_wPoly6Kernel(Vector3s pi, Vector3s pj, scalar H);


__device__ __host__ Vector3s naive_wSpikyKernelGrad(Vector3s pi, Vector3s pj, scalar H);

__device__ Vector3s calcGradConstraint(Vector3s pi, Vector3s pj, scalar h, scalar fp_mass, scalar p0);

__device__ Vector3s calcGradConstraintAtI(int p, Vector3s* d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, int max_neigh, scalar h, scalar fp_mass, scalar p0);

__global__ void naive_initializePositions(Vector3s *d_pos, FluidVolume* g_volumes,
                                     int num_particles, int num_volumes);

__global__ void naive_kupdateVBO(float *vbo, Vector3s *d_pos, int num_particles);
 
__global__ void naive_updateFromForce(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, scalar fp_mass, scalar dt, Vector3s force, int num_particles);

__global__ void naive_updateValForReals(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, scalar dt, int num_particles);  

__global__ void updateXSPHAndOmega(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_omega, int *d_grid, int *d_gridCount, int *d_gridInd, scalar h, int num_particles, int max_neigh);

__global__ void applyVorticity(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_omega, int *d_grid, int *d_gridCount, int *d_gridInd, scalar dt, scalar fp_mass, int num_particles, int max_neigh);

__global__ void preserveFluidBoundaryWithUpdate(Vector3s* d_pos, Vector3s* d_ppos, Vector3s* d_dpos, int num_particles); 

__global__ void preserveFluidBoundary(Vector3s* d_pos, Vector3s* d_ppos, Vector3s* d_dpos, int num_particles); 

__global__ void naive_buildGrid(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, int num_particles, int max_neigh); 

__global__ void applydPToPPos(Vector3s* d_ppos, Vector3s* d_dpos, int num_particles);


__global__ void calcPressures(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_pcalc, int num_particles, int max_neigh, scalar fp_mass, scalar p0, scalar h); 

__global__ void calcLambdas(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_pcalc, scalar *d_lambda, int num_particles, int max_neigh, scalar p0, scalar h, scalar fp_mass); 

__global__ void calcdPos(Vector3s *d_ppos, Vector3s *d_dpos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_lambda, int num_particles, int max_neigh, scalar h, scalar p0, scalar fp_mass); 




 
void naive_initGPUFluid(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd, 
                        int max_neigh, 
                        FluidVolume* h_volumes, int num_volumes,
                        FluidBoundingBox* h_boundingBox,
                        scalar h){

    int num_particles = 0;
    for(int i = 0; i < num_volumes; i++){
        num_particles += h_volumes[i].m_numParticles;
    }

    FluidVolume* g_volumes;
    GPU_CHECKERROR(cudaMalloc((void **)&g_volumes, sizeof(FluidVolume)*num_volumes));
    GPU_CHECKERROR(cudaMemcpy((void *)g_volumes, (void *)h_volumes, sizeof(FluidVolume)*num_volumes, cudaMemcpyHostToDevice));

    GPU_CHECKERROR(cudaMalloc((void **)d_pos, num_particles * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)d_vel, num_particles * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)d_ppos, num_particles * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)d_dpos, num_particles * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaMalloc((void **)d_pcalc, num_particles * sizeof(scalar)));
    GPU_CHECKERROR(cudaMalloc((void **)d_lambda, num_particles * sizeof(scalar)));

    #if naive_VORTICITY > 0 
    GPU_CHECKERROR(cudaMalloc((void **)&d_omega, num_particles * sizeof(Vector3s))); 
    #endif

    int gridSize = ceil(num_particles / (1.0*naive_BLOCKSIZE_1D));
    naive_initializePositions<<<gridSize, naive_BLOCKSIZE_1D>>>(*d_pos, g_volumes, num_particles, num_volumes); 
    GPU_CHECKERROR(cudaGetLastError()); 

    //setup bounding box constants
    scalar h_minX = h_boundingBox->minX();
    scalar h_maxX = h_boundingBox->maxX();
    scalar h_minY = h_boundingBox->minY();
    scalar h_maxY = h_boundingBox->maxY();
    scalar h_minZ = h_boundingBox->minZ();
    scalar h_maxZ = h_boundingBox->maxZ();

    GPU_CHECKERROR(cudaMemcpyToSymbol(c_minX, &h_minX,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_maxX, &h_maxX,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_minY, &h_minY,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_maxY, &h_maxY,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_minZ, &h_minZ,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_maxZ, &h_maxZ,
                                      sizeof(scalar)));
    // figure out dimensions of grid
    int gridSizeX, gridSizeY, gridSizeZ;
    naive_getGridSize(h_boundingBox, h, gridSizeX, gridSizeY, gridSizeZ);

    GPU_CHECKERROR(cudaMemcpyToSymbol(c_gridSizeX, &gridSizeX,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_gridSizeY, &gridSizeY,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_gridSizeZ, &gridSizeZ,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_h, &h,
                                      sizeof(scalar)));


    GPU_CHECKERROR(cudaMalloc((void **)d_grid, gridSizeX * gridSizeY * gridSizeZ * max_neigh * sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)d_gridCount, gridSizeX * gridSizeY * gridSizeZ *sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)d_gridInd, 3 * num_particles * sizeof(int)));

    GPU_CHECKERROR(cudaMemset((void *)*d_vel, 0, num_particles * sizeof(Vector3s)));


    GPU_CHECKERROR(cudaThreadSynchronize());
    GPU_CHECKERROR(cudaFree(g_volumes));

    std::cout << "Done fluid init on gpu" << std::endl;
}

void naive_stepFluid(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, Vector3s *d_dpos, Vector3s *d_omega, 
                      scalar *d_pcalc, scalar *d_lambda, scalar fp_mass,
                      int num_particles, int max_neigh, int *d_grid, int *d_gridCount, int *d_gridInd,
                      int iters, scalar p0,  
                      FluidBoundingBox* h_boundingBox,
                      scalar h,
                      Vector3s accumForce,
                      scalar dt) {

    if(!naive_deviceHappy)
        return;
    
    int gridSizeX, gridSizeY, gridSizeZ;
    naive_getGridSize(h_boundingBox, h, gridSizeX, gridSizeY, gridSizeZ);
    //int grid_size = gridSizeX * gridSizeY * gridSizeZ;

    // update predicted values from forces
    int gridSize = ceil(num_particles / (1.0 * naive_BLOCKSIZE_1D)); 
 
    naive_updateFromForce<<<gridSize, naive_BLOCKSIZE_1D>>>(d_pos, d_vel, d_ppos, fp_mass, dt, accumForce, num_particles); 
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());
 
    GPU_CHECKERROR(cudaMemset((void *)d_dpos, 0, num_particles * sizeof(Vector3s)));
    GPU_CHECKERROR(cudaThreadSynchronize());
 
    // preserve boundary
    preserveFluidBoundaryWithUpdate<<<gridSize, naive_BLOCKSIZE_1D>>>(d_pos, d_ppos, d_dpos, num_particles); 
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());
 

    GPU_CHECKERROR(cudaMemset((void *)d_grid, -1, gridSizeX*gridSizeY*gridSizeZ*max_neigh*sizeof(int)));
    GPU_CHECKERROR(cudaMemset((void *)d_gridCount, 0, gridSizeX * gridSizeY * gridSizeZ *sizeof(int)));

    naive_buildGrid<<<gridSize, naive_BLOCKSIZE_1D>>>(d_ppos, d_grid, d_gridCount, d_gridInd, num_particles, max_neigh);
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());

    for(int loop = 0; loop < iters; ++loop){
        //calculate pressure
        calcPressures<<<gridSize, naive_BLOCKSIZE_1D>>>(d_ppos, d_grid, d_gridCount, d_gridInd, d_pcalc, num_particles, max_neigh, fp_mass, p0, h); 
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaThreadSynchronize());

        // calculate lambda
        calcLambdas<<<gridSize, naive_BLOCKSIZE_1D>>>(d_ppos, d_grid, d_gridCount, d_gridInd, d_pcalc, d_lambda, num_particles, max_neigh, p0, h, fp_mass); 
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaThreadSynchronize());


        // calculate dpos


        calcdPos<<<gridSize, naive_BLOCKSIZE_1D>>>(d_ppos, d_dpos, d_grid, d_gridCount, d_gridInd, d_lambda, num_particles, max_neigh, h, p0, fp_mass); 
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaThreadSynchronize());



        // Preserve boundary
        preserveFluidBoundaryWithUpdate<<<gridSize, naive_BLOCKSIZE_1D>>>(d_pos, d_ppos, d_dpos, num_particles); 
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaThreadSynchronize());
 
        // Apply dp to ppos
        applydPToPPos<<<gridSize, naive_BLOCKSIZE_1D>>>(d_ppos, d_dpos, num_particles);
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaThreadSynchronize());
            

    }

    
    naive_updateValForReals<<<gridSize, naive_BLOCKSIZE_1D>>>(d_pos, d_vel, d_ppos, dt, num_particles); 
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());

    // adjust velocity to taste
    #if naive_XSPH == 0
    return;
    #endif

    updateXSPHAndOmega<<<gridSize, naive_BLOCKSIZE_1D>>>(d_pos, d_vel, d_omega, d_grid, d_gridCount, d_gridInd, h, num_particles, max_neigh);

    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());
    
    #if naive_VORTICITY == 0
    return;
    #endif
    applyVorticity<<<gridSize, naive_BLOCKSIZE_1D>>>(d_pos, d_vel, d_omega, d_grid, d_gridCount, d_gridInd, dt, fp_mass, num_particles, max_neigh);
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());
}

void naive_updateVBO(float *vboptr, Vector3s *d_pos, int num_particles){
    int gridSize = ceil(num_particles / (naive_BLOCKSIZE_1D*1.0));
    naive_kupdateVBO <<< gridSize, naive_BLOCKSIZE_1D>>> (vboptr, d_pos, num_particles);
    cudaError_t err= cudaGetLastError();
    if(err != cudaSuccess){
        naive_deviceHappy = false;
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return; 
    }
    else{
        naive_deviceHappy = true;
    }
    GPU_CHECKERROR(cudaDeviceSynchronize());

    

}

void naive_cleanUp(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd){

    GPU_CHECKERROR(cudaFree(*d_pos));
    GPU_CHECKERROR(cudaFree(*d_vel));
    GPU_CHECKERROR(cudaFree(*d_ppos));
    GPU_CHECKERROR(cudaFree(*d_dpos));
    #if naive_VORTICITY > 0
    GPU_CHECKERROR(cudaFree(*d_omega));
    #endif
    GPU_CHECKERROR(cudaFree(*d_pcalc));
    GPU_CHECKERROR(cudaFree(*d_lambda));

    GPU_CHECKERROR(cudaFree(*d_grid));
    GPU_CHECKERROR(cudaFree(*d_gridCount));
    GPU_CHECKERROR(cudaFree(*d_gridInd));
    
}









__host__ void naive_getGridSize(FluidBoundingBox* fbox, scalar h, int &gridSizeX, int &gridSizeY, int &gridSizeZ){

    gridSizeX = ceil(fbox->width() / h);
    gridSizeY = ceil(fbox->height() / h);
    gridSizeZ = ceil(fbox->depth() / h);

}

__device__ void naive_getGridLocation(Vector3s pos, int &i, int &j, int &k) {

    scalar x = pos.x;
    scalar y = pos.y;
    scalar z = pos.z;

    i = (x - c_minX) / c_h;
    j = (y - c_minY) / c_h;
    k = (z - c_minZ) / c_h;
}

__device__ int naive_getGridIndex(int i, int j, int k) {
    return (c_gridSizeX * c_gridSizeY * k) + (c_gridSizeX * j) + i;
}

__device__ void naive_getGridLocationFromIndex(int id, int &i, int &j, int &k) {
    i = id % c_gridSizeX;
    j = (id / c_gridSizeX) % c_gridSizeY;
    k = (id / c_gridSizeX / c_gridSizeY) % c_gridSizeZ;
}

__device__ Vector3s naive_getFluidVolumePosition(FluidVolume& volume, int k) {

    if (volume.m_mode == kFLUID_VOLUME_MODE_BOX) {
        //random mode not supported
        int xlen = (volume.m_maxX - volume.m_minX) / volume.m_dens_cbrt;
        int ylen = (volume.m_maxY - volume.m_minY) / volume.m_dens_cbrt;
        int zlen = (volume.m_maxZ - volume.m_minZ) / volume.m_dens_cbrt;

        int xindex = (k / zlen / ylen) % xlen;
        int yindex = (k / zlen) % ylen;
        int zindex = k % zlen;

        // printf("%f - %d, %d, %d\n", volume.m_dens_cbrt, xindex, yindex, zindex);

        scalar x = xindex * volume.m_dens_cbrt;
        scalar y = yindex * volume.m_dens_cbrt;
        scalar z = zindex * volume.m_dens_cbrt;
        return Vector3s(x, y, z);
    }
    // sphere mode not supported
    return Vector3s(0, 0, 0);
}

 
__global__ void naive_initializePositions(Vector3s *d_pos, FluidVolume* g_volumes,
                                     int num_particles, int num_volumes) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_particles)
        return;

    int volume_index = -1;
    int offset = 0;
    int volume_size = 0;
    do {
        volume_index++;
        offset += volume_size;
        volume_size = g_volumes[volume_index].m_numParticles;
    } while (offset + volume_size < gid);

    FluidVolume& volume = g_volumes[volume_index];

    d_pos[gid] = naive_getFluidVolumePosition(volume, gid-offset);
}

__global__ void naive_kupdateVBO(float *vbo, Vector3s *d_pos, int num_particles){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < num_particles){
        vbo[id*4+0] = d_pos[id][0];
        vbo[id*4+1] = d_pos[id][1];
        vbo[id*4+2] = d_pos[id][2];
        vbo[id*4+3] = 1.0f;
    }
}

__global__ void naive_updateFromForce(Vector3s* d_pos, Vector3s* d_vel, Vector3s* d_ppos, scalar fp_mass, scalar dt, Vector3s force, int num_particles){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < num_particles){
        d_vel[id] += force * dt / fp_mass;
        //d_vel[id] += force * dt;
        d_ppos[id] = d_pos[id] + d_vel[id]*dt; 
    }
}


__global__ void naive_updateValForReals(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, scalar dt, int num_particles){  
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < num_particles){
        d_vel[id] = (d_ppos[id] - d_pos[id])/dt;
        d_pos[id] = d_ppos[id];
    }
}

__global__ void updateXSPHAndOmega(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_omega, int *d_grid, int *d_gridCount, int *d_gridInd, scalar h, int num_particles, int max_neigh){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p >= num_particles)
        return;
    
    Vector3s dv(0.0, 0.0, 0.0);
    Vector3s vi = d_vel[p];
    Vector3s pi = d_pos[p];
    Vector3s pj; 
    int gi; 
    int q; 
    Vector3s vij;

    #if naive_VORTICITY > 0
    Vector3s omega(0.0, 0.0, 0.0); 
    #endif
    

    for(int i = max(0, d_gridInd[p*3]-1); i <= min(c_gridSizeX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(c_gridSizeY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(c_gridSizeZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = naive_getGridIndex(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    q = d_grid[gi * max_neigh + n];
                    vij = d_vel[q]-vi;
                    pj = d_pos[q]; 
                    dv += vij * naive_wPoly6Kernel(pi, pj, h);    

                    #if naive_VORTICITY > 0
                    omega += glm::cross(vij, naive_wSpikyKernelGrad(pi, pj, h)); 
                    #endif          
                }
            }
        }
    }

    dv *= naive_C;
    d_vel[p] += dv; 
    #if naive_VORTICITY > 0
    d_omega[p] = omega;
    #endif
}

__global__ void applyVorticity(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_omega, int *d_grid, int *d_gridCount, int *d_gridInd, scalar dt, scalar fp_mass, int num_particles, int max_neigh){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p >= num_particles)
        return;
   
    Vector3s pi = d_pos[p];
    Vector3s pj; 
    Vector3s omega = d_omega[p]; 
    scalar dom;
    int gi;
    int q;
    Vector3s vort(0, 0, 0); 
    Vector3s dp;
    

    for(int i = max(0, d_gridInd[p*3]-1); i <= min(c_gridSizeX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(c_gridSizeY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(c_gridSizeZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = naive_getGridIndex(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    q = d_grid[gi * max_neigh + n];
                    pj = d_pos[q]; 
                    dp = pj - pi; 
                    dom = glm::length(omega - d_omega[q]);     
                    vort[0] += dom / (dp[0]+.001);
                    vort[1] += dom / (dp[1]+.001);
                    vort[2] += dom / (dp[2]+.001);                      
                }
            }
        }
    }
    vort /= (glm::length(vort) + .001);    
    d_vel[p] += (scalar)(dt * naive_VORT_EPS / fp_mass) * (glm::cross(vort, omega)); 
}

__global__ void preserveFluidBoundaryWithUpdate(Vector3s* d_pos, Vector3s* d_ppos, Vector3s* d_dpos, int num_particles){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i >= num_particles){
        return;
    }
    scalar pposX = d_ppos[i][0] + d_dpos[i][0];
    scalar pposY = d_ppos[i][1] + d_dpos[i][1];
    scalar pposZ = d_ppos[i][2] + d_dpos[i][2]; 

    scalar shift = (i%124 )* 1.0/10000.0; 
    if(pposX < c_minX + naive_EPS){
        d_dpos[i][0] = c_minX + naive_EPS + shift - d_ppos[i][0];
        d_ppos[i][0] += d_dpos[i][0];
    }
    else if(pposX > c_maxX - naive_EPS){ 
        d_dpos[i][0] = c_maxX - naive_EPS - shift- d_ppos[i][0]; 
        d_ppos[i][0] += d_dpos[i][0];
    }
    if(pposY < c_minY + naive_EPS){
        d_dpos[i][1] = c_minY + naive_EPS + shift  - d_ppos[i][1];
        d_ppos[i][1] += d_dpos[i][1];
    }
    else if(pposY > c_maxY - naive_EPS){
        d_dpos[i][1] = c_maxY - naive_EPS - shift - d_ppos[i][1];
        d_ppos[i][1] += d_dpos[i][1];
    }
    if(pposZ < c_minZ + naive_EPS){
        d_dpos[i][2] = c_minZ + naive_EPS + shift - d_ppos[i][2];
        d_ppos[i][2] += d_dpos[i][2];
    }
    else if(pposZ > c_maxZ - naive_EPS){
        d_dpos[i][2] = c_maxZ - naive_EPS - shift - d_ppos[i][2];
        d_ppos[i][2] += d_dpos[i][2];
    }
}

__global__ void preserveFluidBoundary(Vector3s *d_pos, Vector3s *d_ppos, Vector3s *d_dpos, int num_particles){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i >= num_particles){
        return;
    }
    scalar pposX = d_ppos[i][0] + d_dpos[i][0];
    scalar pposY = d_ppos[i][1] + d_dpos[i][1];
    scalar pposZ = d_ppos[i][2] + d_dpos[i][2]; 
    scalar shift = (i%124 )* 1.0/10000.0; 
 
    if(pposX < c_minX + naive_EPS){
        d_dpos[i][0] = c_minX + naive_EPS + shift - d_ppos[i][0];
    }
    else if(pposX > c_maxX - naive_EPS){
        d_dpos[i][0] = c_maxX - naive_EPS - shift - d_ppos[i][0]; 
    }
    if(pposY < c_minY + naive_EPS){
        d_dpos[i][1] = c_minY + naive_EPS + shift - d_ppos[i][1];
    }
    else if(pposY > c_maxY - naive_EPS){
        d_dpos[i][1] = c_maxY - naive_EPS - shift - d_ppos[i][1];
    }
    if(pposZ < c_minZ + naive_EPS){
        d_dpos[i][2] = c_minZ + naive_EPS + shift - d_ppos[i][2];
    }
    else if(pposZ > c_maxZ - naive_EPS){
        d_dpos[i][2] = c_maxZ - naive_EPS - shift - d_ppos[i][2];
    }
}

__global__ void naive_buildGrid(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, int num_particles, int max_neigh){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id >= num_particles){
        return;
    }
    
    int gx; 
    int gy; 
    int gz; 

    naive_getGridLocation(d_ppos[id], gx, gy, gz);
    int gid = naive_getGridIndex(gx, gy, gz);
    d_gridInd[id * 3] = gx;
    d_gridInd[id * 3+1] = gy;
    d_gridInd[id * 3+2] = gz;

    int actgid = gid * max_neigh + d_gridCount[gid];
      
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
    //d_grid[gid * max_neigh + d_gridCount[gid]] = id;
    atomicAdd(&d_gridCount[gid], 1);
}

__global__ void applydPToPPos(Vector3s* d_ppos, Vector3s* d_dpos, int num_particles){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < num_particles){
        d_ppos[id] += d_dpos[id];
        //d_ppos[id] += Vector3s(1.0, 0, 0);
    }
}

__global__ void calcPressures(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_pcalc, int num_particles, int max_neigh, scalar fp_mass, scalar p0, scalar h){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p >= num_particles){
        return;
    }
    scalar press = 0;
    int ncount = 0;
    int gi;
    Vector3s pi = d_ppos[p]; 
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(c_gridSizeX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(c_gridSizeY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(c_gridSizeZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = naive_getGridIndex(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    scalar pressN = naive_wPoly6Kernel(pi, d_ppos[d_grid[gi * max_neigh + n]], h); 
                    press += pressN;
                    if(pressN > 0)
                        ++ ncount; 

                }
            }
        }
    }     
    if(ncount <= naive_MIN_NEIGHBORS && d_pcalc[p] == 0){ // don't count self
        d_pcalc[p] = p0; 

   }
    else {
        d_pcalc[p] = fp_mass * press; // Wow I totally forgot that
    }

}

__global__ void calcLambdas(Vector3s *d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_pcalc, scalar *d_lambda, int num_particles, int max_neigh, scalar p0, scalar h, scalar fp_mass){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p >= num_particles)
        return;

    scalar top = -(d_pcalc[p]/p0 - 1.0);
    scalar gradSum = 0;
    scalar gradL = 0;
    Vector3s pi = d_ppos[p];


    int gi;
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(c_gridSizeX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(c_gridSizeY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(c_gridSizeZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = naive_getGridIndex(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    gradL = glm::length(calcGradConstraint(pi, d_ppos[d_grid[gi * max_neigh + n]], h, fp_mass, p0));
                    gradSum += gradL * gradL;
                }
            }
        }
    }
     
    gradL = glm::length(calcGradConstraintAtI(p, d_ppos, d_grid, d_gridCount, d_gridInd, max_neigh, h, fp_mass, p0));
    gradSum += gradL * gradL;
    d_lambda[p] = top / (gradSum + naive_EPS);
} 

 
 
__global__ void calcdPos(Vector3s *d_ppos, Vector3s *d_dpos, int *d_grid, int *d_gridCount, int *d_gridInd, scalar *d_lambda, int num_particles, int max_neigh, scalar h, scalar p0, scalar fp_mass){
    int p = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(p >= num_particles)
        return;

    Vector3s dp(0.0, 0.0, 0.0);
    int q = 0; 
    int gi = 0;
    scalar plambda = d_lambda[p]; 
    Vector3s pi = d_ppos[p];
    Vector3s pj; 

    scalar scorr = 0; // bla 

    for(int i = max(0, d_gridInd[p*3]-1); i <= min(c_gridSizeX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(c_gridSizeY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(c_gridSizeZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = naive_getGridIndex(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    q = d_grid[gi * max_neigh + n];
                    pj = d_ppos[q];                
    
                #if naive_ART_PRESSURE > 0
                    scalar top = naive_wPoly6Kernel(pi, pj, h); 
                    scorr = - naive_K * (pow(top / naive_QSCALE, naive_N)); 
                #endif

                    dp += (plambda + d_lambda[q] + scorr) * naive_wSpikyKernelGrad(pi, pj, h);
                }
            }
        }
    }
    d_dpos[p] = fp_mass * dp / p0;
    //d_dpos[p] = Vector3s(.1, 0, 0);
}



__device__ Vector3s calcGradConstraint(Vector3s pi, Vector3s pj, scalar h, scalar fp_mass, scalar p0){
    return (fp_mass*naive_wSpikyKernelGrad(pi, pj, h)/(scalar(- p0))); 
}

__device__ Vector3s calcGradConstraintAtI(int p, Vector3s* d_ppos, int *d_grid, int *d_gridCount, int *d_gridInd, int max_neigh, scalar h, scalar fp_mass, scalar p0){
    Vector3s sumGrad(0.0, 0.0, 0.0);
    Vector3s pi = d_ppos[p]; 
    int gi; 
    for(int i = max(0, d_gridInd[p*3]-1); i <= min(c_gridSizeX-1, d_gridInd[p*3]+1); ++i){
        for(int j = max(0, d_gridInd[p*3+1]-1); j <= min(c_gridSizeY-1, d_gridInd[p*3+1]+1); ++j){
            for(int k = max(0, d_gridInd[p*3+2]-1); k <= min(c_gridSizeZ-1, d_gridInd[p*3+2]+1); ++k){
                gi = naive_getGridIndex(i, j, k);
                for(int n = 0; n < d_gridCount[gi]; ++n){ // for all particles in the grid
                    sumGrad += naive_wSpikyKernelGrad(pi, d_ppos[d_grid[gi * max_neigh + n]], h);
                }
            }
        }
    }     
    return fp_mass*sumGrad / p0; 

}

__device__ __host__ scalar naive_wPoly6Kernel(Vector3s pi, Vector3s pj, scalar H){
    scalar r = glm::distance(pi, pj); 
    if(r > H || r < 0)
        return 0; 

    r = ((H * H) - (r * r)); 
    r = r * r * r; // (h^2 - r^2)^3
    return r * (315.0 / (64.0 * PI * H * H * H * H * H * H * H * H * H));

}

__device__ __host__ Vector3s naive_wSpikyKernelGrad(Vector3s pi, Vector3s pj, scalar H){
    Vector3s dp = pi - pj; 
    scalar r = glm::length(dp);  
    if(r > H || r < 0)
        return Vector3s(0.0, 0.0, 0.0); 
    scalar scale = 45.0 / (PI * H * H * H * H * H * H) * (H - r) * (H - r); 
    return scale * dp / (r+.0001f); 
//    return scale * dp; 
}



#endif
