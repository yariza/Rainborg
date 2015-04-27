#ifdef GPU_ENABLED
#include "GridGPUFluidKernel.h"
#include "../FluidVolume.h"
#include "GPUHelper.h"

const int kgrid_BLOCKSIZE_1D = 512;
const int kgrid_BLOCKSIZE_3D = 8;

bool grid_deviceHappy = false;

__device__ Vector3s kgrid_getFluidVolumePosition(FluidVolume& volume, int k);

__global__ void kgrid_initializePositions(Vector3s *g_pos, FluidVolume* g_volumes,
                                     int num_particles, int num_volumes);
__global__ void kgrid_updateVBO(float* dptrvert, Vector3s *g_pos, int num_particles);

void grid_initGPUFluid(Vector3s *g_pos, Vector3s *g_vel,
                       int *g_neighbors, int *g_gridIndex,
                       FluidVolume* h_volumes, int num_volumes) {

    int num_particles = 0;
    for (int i=0; i<num_volumes; i++) {
        num_particles += h_volumes[i].m_numParticles;
    }

    // initialize volumes array (free afterward)
    FluidVolume* g_volumes;
    GPU_CHECKERROR(cudaMalloc((void **)&g_volumes,
                              sizeof(FluidVolume)*num_volumes));
    GPU_CHECKERROR(cudaMemcpy((void *)g_volumes, (void *)h_volumes,
                              sizeof(FluidVolume)*num_volumes,
                              cudaMemcpyHostToDevice));

    // allocate position and velocity array
    GPU_CHECKERROR(cudaMalloc((void **)&g_pos,
                              sizeof(Vector3s)*num_particles));
    GPU_CHECKERROR(cudaMalloc((void **)&g_vel,
                              sizeof(Vector3s)*num_particles));
    // set velocities to 0
    GPU_CHECKERROR(cudaMemset((void *)g_vel, 0,
                              sizeof(Vector3s)*num_particles));

    int gridSize = ceil(num_particles / (kgrid_BLOCKSIZE_1D*1.0));
    kgrid_initializePositions <<< gridSize, kgrid_BLOCKSIZE_1D
                              >>> (g_pos, g_volumes, num_particles, num_volumes);
    GPU_CHECKERROR(cudaGetLastError());
    GPU_CHECKERROR(cudaThreadSynchronize());

    cudaFree(g_volumes); // we don't use this anymore
}

void grid_updateVBO(float *vboptr, Vector3s *g_pos, int num_particles) {

    if (vboptr == NULL) {
        printf("oh no!!\n");
    }
    int gridSize = ceil(num_particles / (kgrid_BLOCKSIZE_1D*1.0));
    kgrid_updateVBO <<< gridSize, kgrid_BLOCKSIZE_1D
                    >>> (vboptr, g_pos, num_particles);

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

__global__ void kgrid_initializePositions(Vector3s *g_pos, FluidVolume* g_volumes,
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
    g_pos[gid] = kgrid_getFluidVolumePosition(volume, gid - offset);
}

__global__ void kgrid_updateVBO(float* vbo, Vector3s *g_pos, int num_particles) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_particles) {
        vbo[gid*4+0] = g_pos[gid][0];
        vbo[gid*4+1] = g_pos[gid][1];
        vbo[gid*4+2] = g_pos[gid][2];
        vbo[gid*4+3] = 1.0f;
    }
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

        scalar x = xindex * volume.m_dens_cbrt;
        scalar y = yindex * volume.m_dens_cbrt;
        scalar z = zindex * volume.m_dens_cbrt;
        return Vector3s(x, y, z);
    }
    // sphere mode not supported
    return Vector3s(0, 0, 0);
}


#endif
