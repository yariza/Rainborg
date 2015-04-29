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
const int naive_BLOCKSIZE_REDUCED = 256;
const int naive_MAX_NEIGHBORS = 10;
const int naive_MIN_NEIGHBORS = 2;

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




void naive_initGPUFluid(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd, 
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

    #if VORTICITY > 0 
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


    GPU_CHECKERROR(cudaMalloc((void **)d_grid, gridSizeX * gridSizeY * gridSizeZ * naive_MAX_NEIGHBORS * sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)d_gridCount, gridSizeX * gridSizeY * gridSizeZ *sizeof(int)));
    GPU_CHECKERROR(cudaMalloc((void **)d_gridInd, 3 * num_particles * sizeof(int)));

    GPU_CHECKERROR(cudaMemset((void *)*d_vel, 0, num_particles * sizeof(Vector3s)));


    GPU_CHECKERROR(cudaThreadSynchronize());
    GPU_CHECKERROR(cudaFree(g_volumes));

    std::cout << "Done fluid init on gpu" << std::endl;
}

void naive_stepFluid(
                      int num_particles,
                      FluidBoundingBox* h_boundingbox,
                      scalar h,
                      Vector3s accumForce,
                      scalar dt) {


}

void naive_updateVBO(float *vboptr, Vector3s *d_pos, int num_particles){
    


    

}

void naive_cleanUp(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd){

    GPU_CHECKERROR(cudaFree(*d_pos));
    GPU_CHECKERROR(cudaFree(*d_vel));
    GPU_CHECKERROR(cudaFree(*d_ppos));
    GPU_CHECKERROR(cudaFree(*d_dpos));
    #if VORTICITY > 0
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




#endif
