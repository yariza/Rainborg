#ifdef GPU_ENABLED
#include "GridGPUFluid.h"
#include "gpu/GridGPUFluidKernel.h"
#include "FluidSimpleGravityForce.h"
#include "Scene.h"

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

  Vector3s accumForce = Vector3s(0, 0, 0);

  std::vector<FluidForce*> &forces = scene.getFluidForces();
  for (std::vector<FluidForce*>::size_type i=0;
       i < forces.size(); i++) {

    FluidForce *force = forces[i];

    // check if simple gravity force
    FluidSimpleGravityForce *gravityForce = dynamic_cast<FluidSimpleGravityForce*>(force);

    if (gravityForce) {
      accumForce += gravityForce->getGlobalForce();
    }
  }

  grid_stepFluid(&d_neighbors, &d_gridIndex,
                 &d_grid,
                 &d_particles,
                 getNumParticles(),
                 m_boundingBox, m_h,
                 accumForce,
                 dt);
}

void GridGPUFluid::loadFluidVolumes() {

  FluidVolume h_volumes[m_volumes.size()];
  for (int i=0; i<m_volumes.size(); i++) {
    h_volumes[i] = m_volumes[i];
  }

  grid_initGPUFluid(&d_neighbors, &d_gridIndex,
                    &d_grid,
                    &d_particles,
                    h_volumes, m_volumes.size(),
                    m_boundingBox, m_h);


}

void GridGPUFluid::updateVBO(float* dptrvert) {
  grid_updateVBO(dptrvert, d_particles, getNumParticles());
}
#endif
