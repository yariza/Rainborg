#include "GridGPUFluid.h"
#include "gpu/GridGPUFluidKernel.h"
#include <iostream>
GridGPUFluid::GridGPUFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors) 
: Fluid(mass, p0, h, iters, maxNeighbors, minNeighbors)
, m_eps(0.01)
{
}

GridGPUFluid::GridGPUFluid(GridGPUFluid& otherFluid)
: Fluid(otherFluid)
, m_eps(0.01)
{

}

GridGPUFluid::~GridGPUFluid() {

}

void GridGPUFluid::stepSystem(Scene& scene, scalar dt) {

}

void GridGPUFluid::loadFluidVolumes() {

  FluidVolume h_volumes[m_volumes.size()];
  for (int i=0; i<m_volumes.size(); i++) {
    h_volumes[i] = m_volumes[i];
  }

  grid_initGPUFluid(&d_pos, &d_vel,
                    &d_neighbors, &d_gridIndex,
                    h_volumes, m_volumes.size());


}

void GridGPUFluid::updateVBO(float* dptrvert) {
  grid_updateVBO(dptrvert, d_pos, getNumParticles());
}
