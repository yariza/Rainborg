#ifdef GPU_ENABLED
#include "GridGPUFluidKernel.h"
#include "GPUHelper.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

const int kgrid_BLOCKSIZE_1D = 512;
const int kgrid_BLOCKSIZE_3D = 8;
const int kgrid_NUM_NEIGHBORS = 5;
const int kgrid_MAX_CELL_SIZE = 50;

bool grid_deviceHappy = true;

// fluid characteristics
__constant__ scalar c_h;

// grid bounds
__constant__ scalar c_minX;
__constant__ scalar c_maxX;
__constant__ scalar c_minY;
__constant__ scalar c_maxY;
__constant__ scalar c_minZ;
__constant__ scalar c_maxZ;
__constant__ scalar c_eps;

// grid size
__constant__ int c_gridSizeX;
__constant__ int c_gridSizeY;
__constant__ int c_gridSizeZ;

///////////////////////////////////////////////
/// Function Headers
///////////////////////////////////////////////

// helper functions for getting grid indices
__device__ void kgrid_getGridLocation(Vector3s pos, int &i, int &j, int &k);
__device__ int kgrid_getGridIndex(int i, int j, int k);

// helper function for getting grid size
__host__ void hgrid_getGridSize(FluidBoundingBox* fbox, scalar h,
                               int &gridSizeX, int &gridSizeY, int &gridSizeZ);

__device__ Vector3s kgrid_getFluidVolumePosition(FluidVolume& volume, int k);

__global__ void kgrid_initializePositions(grid_gpu_block_t *g_particles, FluidVolume* g_volumes,
                                     int num_particles, int num_volumes);
__global__ void kgrid_updateVBO(float* vbo, grid_gpu_block_t *g_particles, int num_particles);


// apply forces
__global__ void kgrid_applyForces(grid_gpu_block_t *g_particles,
                                  int num_particles,
                                  Vector3s accumForce,
                                  scalar dt);

// clear grid
__global__ void kgrid_clearGrid(int *g_grid, int grid_size);

// get grid indices
__global__ void kgrid_getGridIndices(grid_gpu_block_t *g_particles,
                                     int num_particles,
                                     int *g_gridIndex);

// TODO


// update velocity
__global__ void kgrid_updateVelocity(grid_gpu_block_t *g_particles,
                                     int num_particles,
                                     scalar dt);

// update position
__global__ void kgrid_updatePosition(grid_gpu_block_t *g_particles,
                                     int num_particles);

// Nearest Neighbor kernels
__global__ void kgrid_clearGrid(int *grid, int grid_size);


////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////

/// Init fluid

void grid_initGPUFluid(int **g_neighbors, int **g_gridIndex,
                       int **g_grid,
                       int **g_gridUniqueIndex, int **g_partUniqueIndex,
                       grid_gpu_block_t **g_particles,
                       FluidVolume* h_volumes, int num_volumes,
                       FluidBoundingBox* h_boundingBox,
                       scalar h) {

    int num_particles = 0;
    for (int i=0; i<num_volumes; i++) {
        num_particles += h_volumes[i].m_numParticles;
    }

    // initialize volumes array (free afterward)
    FluidVolume* g_volumes;
    GPU_CHECKERROR(cudaMalloc((void **)&g_volumes,
                              sizeof(FluidVolume)*num_volumes));
    // printf("%f, %f; %f, %f; %f, %f\n", h_volumes[0].m_minX, h_volumes[0].m_maxX, h_volumes[0].m_minY, h_volumes[0].m_maxY, h_volumes[0].m_minZ, h_volumes[0].m_maxZ); 

    GPU_CHECKERROR(cudaMemcpy((void *)g_volumes, (void *)h_volumes,
                              sizeof(FluidVolume)*num_volumes,
                              cudaMemcpyHostToDevice));

    // allocate particles
    GPU_CHECKERROR(cudaMalloc((void **)g_particles,
                              sizeof(grid_gpu_block_t)*num_particles));

    // allocate neighbors array (num_particles * num_neighbors * int)
    GPU_CHECKERROR(cudaMalloc((void **)g_neighbors,
                              sizeof(int)*num_particles*kgrid_NUM_NEIGHBORS));

    // allocate grid index array (num_particles * int)
    GPU_CHECKERROR(cudaMalloc((void **)g_gridIndex,
                              sizeof(int)*num_particles));

    // allocate grid unique index array (initially num_particles * int)
    GPU_CHECKERROR(cudaMalloc((void **)g_gridUniqueIndex,
                              sizeof(int)*num_particles));

    // allocate part unique index array (initially num_particles * int)
    GPU_CHECKERROR(cudaMalloc((void **)g_partUniqueIndex,
                              sizeof(int)*num_particles));


    int gridSize = ceil(num_particles / (kgrid_BLOCKSIZE_1D*1.0));
    kgrid_initializePositions <<< gridSize, kgrid_BLOCKSIZE_1D
                              >>> (*g_particles, g_volumes, num_particles, num_volumes);
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
    hgrid_getGridSize(h_boundingBox, h, gridSizeX, gridSizeY, gridSizeZ);

    GPU_CHECKERROR(cudaMemcpyToSymbol(c_gridSizeX, &gridSizeX,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_gridSizeY, &gridSizeY,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_gridSizeZ, &gridSizeZ,
                                      sizeof(scalar)));
    GPU_CHECKERROR(cudaMemcpyToSymbol(c_h, &h,
                                      sizeof(scalar)));

    // allocate grid
    GPU_CHECKERROR(cudaMalloc((void **)g_grid,
                              sizeof(int)*gridSizeX*gridSizeY*gridSizeZ));

    GPU_CHECKERROR(cudaThreadSynchronize());

    cudaFree(g_volumes); // we don't use this anymore
}

__global__ void kgrid_initializePositions(grid_gpu_block_t *g_particles, FluidVolume* g_volumes,
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

    g_particles[gid].pos = kgrid_getFluidVolumePosition(volume, gid - offset);
    g_particles[gid].vec1 = Vector3s(0, 0, 0); // velocity
}

/// Update VBO

void grid_updateVBO(float *vboptr, grid_gpu_block_t *g_particles, int num_particles) {

    if (vboptr == NULL) {
        printf("oh no!!\n");
    }
    int gridSize = ceil(num_particles / (kgrid_BLOCKSIZE_1D*1.0));
    kgrid_updateVBO <<< gridSize, kgrid_BLOCKSIZE_1D
                    >>> (vboptr, g_particles, num_particles);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        grid_deviceHappy = false;
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        std::cout << "vboptr: " << vboptr << std::endl;
        return;
    }
    else {
        grid_deviceHappy = true;
    }

    GPU_CHECKERROR(cudaDeviceSynchronize());
}

__global__ void kgrid_updateVBO(float* vbo, grid_gpu_block_t *g_particles, int num_particles) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_particles) {
    //if(gid < 10){
        vbo[gid*4+0] = g_particles[gid].pos.x;
        vbo[gid*4+1] = g_particles[gid].pos.y;
        vbo[gid*4+2] = g_particles[gid].pos.z;
        //vbo[gid*4+0] = 5.0f;
        //vbo[gid*4+1] = 5.0f;
        //vbo[gid*4+2] = 5.0f;
        vbo[gid*4+3] = 1.0f;
    }
}

/// Step function

void grid_stepFluid(int **g_neighbors, int **g_gridIndex,
                    int **g_grid,
                    int **g_gridUniqueIndex, int **g_partUniqueIndex,
                    grid_gpu_block_t **g_particles,
                    int num_particles,
                    FluidBoundingBox* h_boundingBox,
                    scalar h,
                    Vector3s accumForce,
                    scalar dt) {

    int gridSizeX, gridSizeY, gridSizeZ;
    hgrid_getGridSize(h_boundingBox, h, gridSizeX, gridSizeY, gridSizeZ);
    int grid_size = gridSizeX*gridSizeY*gridSizeZ;

    int blocksPerParticles = ceil(num_particles / (kgrid_BLOCKSIZE_1D*1.0));
    int blocksPerGridCells = ceil(grid_size / (kgrid_BLOCKSIZE_1D*1.0));

    // step 1: apply forces, predict position
    kgrid_applyForces <<< blocksPerParticles, kgrid_BLOCKSIZE_1D
                      >>> (*g_particles, num_particles, accumForce, dt);
    GPU_CHECKERROR(cudaGetLastError());


    // step 2: find k nearest neighbors

    // step 2a: reset grid
    kgrid_clearGrid <<< blocksPerGridCells, kgrid_BLOCKSIZE_1D
                    >>> (*g_grid, grid_size);
    GPU_CHECKERROR(cudaGetLastError());

    // step 2b: get gridIDs
    kgrid_getGridIndices <<< blocksPerParticles, kgrid_BLOCKSIZE_1D
                         >>> (*g_particles, num_particles, *g_gridIndex);
    GPU_CHECKERROR(cudaGetLastError());

    // step 2c: sort particles by gridID
    thrust::device_ptr<int> t_gridIndex = thrust::device_pointer_cast(*g_gridIndex);
    thrust::device_ptr<grid_gpu_block_t> t_particles =
        thrust::device_pointer_cast(*g_particles);

    thrust::sort_by_key(t_gridIndex, t_gridIndex+num_particles, t_particles);
    GPU_CHECKERROR(cudaGetLastError());

    // step 2d: 

    // step 7: update velocity
    kgrid_updateVelocity <<< blocksPerParticles, kgrid_BLOCKSIZE_1D
                      >>> (*g_particles, num_particles, dt);
    GPU_CHECKERROR(cudaGetLastError());


    // step 9: update position
    kgrid_updatePosition <<< blocksPerParticles, kgrid_BLOCKSIZE_1D
                      >>> (*g_particles, num_particles);
    GPU_CHECKERROR(cudaGetLastError());



    GPU_CHECKERROR(cudaDeviceSynchronize());
}

/// Apply Forces

__global__ void kgrid_applyForces(grid_gpu_block_t *g_particles,
                                  int num_particles,
                                  Vector3s accumForce,
                                  scalar dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_particles) {
        Vector3s pos = g_particles[id].pos;
        Vector3s vel = g_particles[id].vec1;
        vel += dt * accumForce;
        Vector3s ppos = pos + dt * vel;
        g_particles[id].vec1 = vel; //velocity
        g_particles[id].vec2 = ppos; // predicted pos
    }
}

/// Reset Grid

__global__ void kgrid_clearGrid(int *g_grid, int grid_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < grid_size) {
        g_grid[id] = -1; // indicates no particle
    }
}

/// Get Grid Indices

__global__ void kgrid_getGridIndices(grid_gpu_block_t *g_particles,
                                     int num_particles,
                                     int *g_gridIndex) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_particles) {
        Vector3s pos = g_particles[id].pos;
        int i, j, k;
        kgrid_getGridLocation(pos, i, j, k);
        int index = kgrid_getGridIndex(i, j, k);
        g_gridIndex[id] = index;
    }
}


/// update velocity

__global__ void kgrid_updateVelocity(grid_gpu_block_t *g_particles,
                                     int num_particles,
                                     scalar dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_particles) {
        Vector3s pos = g_particles[id].pos; // pos
        Vector3s ppos = g_particles[id].vec2; // ppos
        Vector3s vel = (ppos - pos) / dt;
        g_particles[id].vec1 = vel; //velocity
    }
}


/// update position

__global__ void kgrid_updatePosition(grid_gpu_block_t *g_particles,
                                     int num_particles) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_particles) {
        Vector3s ppos = g_particles[id].vec2; // ppos
        g_particles[id].pos = ppos; //pos = ppos
    }

}



////////////////////////////////////////
/// Helper functions
////////////////////////////////////////

__host__ void hgrid_getGridSize(FluidBoundingBox* fbox, scalar h,
                               int &gridSizeX, int &gridSizeY, int &gridSizeZ) {
    gridSizeX = ceil(fbox->width() / h);
    gridSizeY = ceil(fbox->height() / h);
    gridSizeZ = ceil(fbox->depth() / h);
}

__device__ void kgrid_getGridLocation(Vector3s pos, int &i, int &j, int &k) {

    scalar x = pos.x;
    scalar y = pos.y;
    scalar z = pos.z;

    i = (x - c_minX) / c_h;
    j = (y - c_minY) / c_h;
    k = (z - c_minZ) / c_h;
}

__device__ int kgrid_getGridIndex(int i, int j, int k) {
    return (c_gridSizeX * c_gridSizeY * k) + (c_gridSizeX * j) + i;
}

__device__ Vector3s kgrid_getFluidVolumePosition(FluidVolume& volume, int k) {

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
