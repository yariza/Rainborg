#ifndef __GRID_GPU_FLUID_H__
#define __GRID_GPU_FLUID_H__

#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"
#include "Fluid.h"

class Scene;

class GridGPUFluid : public Fluid {

public:
    GridGPUFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeigh = 20, int minNeighbor = 3);
    GridGPUFluid(GridGPUFluid& otherFluid);
    virtual ~GridGPUFluid();

    virtual void stepSystem(Scene& scene, scalar dt);
    virtual void loadFluidVolumes();

private:
    scalar m_eps;
};

#endif