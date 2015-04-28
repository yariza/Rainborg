#ifndef __GRID_GPU_FLUID_H__
#define __GRID_GPU_FLUID_H__

#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"
#include "Fluid.h"
#include "gpu/GridGPUFluidKernel.h"

class Scene;

class GridGPUFluid : public Fluid {

public:
    GridGPUFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeigh = 20, int minNeighbor = 3);
    GridGPUFluid(GridGPUFluid& otherFluid);
    virtual ~GridGPUFluid();

    virtual void stepSystem(Scene& scene, scalar dt);
    virtual void loadFluidVolumes();
    virtual void updateVBO(float* dptrvert);

private:
    scalar m_eps;

    // device global memory

    // particle array
    grid_gpu_block_t *d_particles;

    // grid neighbors
    // stores neighbor particle ids.
    // dimension num_particles*max_neighbors
    int *d_neighbors;

    // grid index: per particle
    // stores grid index of particle
    int *d_gridIndex;

    // copy of grid index
    // later used to extract unique copies
    // initial size num_particles (resized to grid_size)
    int *d_gridUniqueIndex;

    // stores the first particle id for each unique grid index
    // initial size num_particles (resized to grid_size)
    int *d_particleIndex;

    // grid: stores the gridIndex id of first element in grid location
    // dimension x*y*z
    int *d_grid;

};
#endif
