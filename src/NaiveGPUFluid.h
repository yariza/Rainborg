#ifndef __NAIVE_GPU_FLUID_H__
#define __NAIVE_GPU_FLUID_H__

#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"
#include "Fluid.h"
#include "gpu/NaiveGPUFluidKernel.h"

class Scene;

class NaiveGPUFluid : public Fluid {

public:
    NaiveGPUFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeigh = 20, int minNeighbor = 3);
    NaiveGPUFluid(NaiveGPUFluid& otherFluid);
    virtual ~NaiveGPUFluid();

    virtual void stepSystem(Scene& scene, scalar dt);
    virtual void loadFluidVolumes();
    virtual void updateVBO(float* dptrvert);

private:
    scalar m_eps;
    Vector3s *d_pos;
    Vector3s *d_vel;
    Vector3s *d_ppos;
    Vector3s *d_dpos;
    Vector3s *d_omega;
    scalar *d_pcalc; 
    scalar *d_lambda;
    int *d_grid;
    int *d_gridCount;
    int *d_gridInd;   




};
#endif
