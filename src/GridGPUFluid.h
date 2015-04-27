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

    // device global memory: per particle
    Vector3s *d_pos;
    Vector3s *d_vel;
    // grid neighbors: per grid
    // stores neighbor particle id.
    // dimension x*y*z*max_neighbors
    int *d_neighbors;
    // grid index: per particle
    // stores grid index of particle
    int *d_gridIndex;

};
#endif
