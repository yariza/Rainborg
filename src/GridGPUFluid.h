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
    // dimension num_particles*num_neighbors
    int *d_neighbors;

    // grid index: per particle
    // stores grid index of particle
    int *d_gridIndex;

    // grid: stores the particle id of first element in grid location
    // dimension x*y*z
    int *d_grid;

    Vector4s *d_colors;

};
#endif
