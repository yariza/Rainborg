#ifdef GPU_ENABLED
#include "GridGPUFluidKernel.h"
#include "GPUHelper.h"

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

__device__ Vector3s kgrid_getFluidVolumePosition(FluidVolume& volume, int k);

__global__ void kgrid_initializePositions(grid_gpu_block_t *g_particles, FluidVolume* g_volumes,
                                     int num_particles, int num_volumes);
__global__ void kgrid_updateVBO(float* vbo, grid_gpu_block_t *g_particles, int num_particles);

////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////

/// Init fluid

void grid_initGPUFluid(int **g_neighbors, int **g_gridIndex,
                       int **g_grid,
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
    int gridSizeX = ceil(h_boundingBox->width() / h);
    int gridSizeY = ceil(h_boundingBox->height() / h);
    int gridSizeZ = ceil(h_boundingBox->depth() / h);

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

    GPU_CHECKERROR(cudaThreadSynchronize());
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
                    grid_gpu_block_t **g_particles,
                    int num_particles,
                    Vector3s accumForce,
                    scalar dt) {

    
}

/// Setup Grid





////////////////////////////////////////
/// Device functions
////////////////////////////////////////

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
