#ifdef GPU_ENABLED
#include "NaiveGPUFluid.h"
#include "gpu/NaiveGPUFluidKernel.h"
#include "FluidSimpleGravityForce.h"
#include "Scene.h"

#include <iostream>
NaiveGPUFluid::NaiveGPUFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors) 
: Fluid(mass, p0, h, iters, maxNeighbors, minNeighbors)
, m_eps(0.01)
{
}

NaiveGPUFluid::NaiveGPUFluid(NaiveGPUFluid& otherFluid)
: Fluid(otherFluid)
, m_eps(0.01)
{

}

NaiveGPUFluid::~NaiveGPUFluid() {
    // clean up
    naive_cleanUp(&d_pos, &d_vel, &d_ppos, &d_dpos, &d_omega, 
                        &d_pcalc, &d_lambda, &d_grid, &d_gridCount, &d_gridInd); 
}

void NaiveGPUFluid::stepSystem(Scene& scene, scalar dt) {

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

    naive_stepFluid(d_pos, d_vel, d_ppos, d_dpos, m_fpmass,
                      getNumParticles(), 
                      m_boundingBox,
                      m_h,
                      accumForce,
                      dt);
 
    /*
  naive_stepFluid(&d_neighbors, &d_gridIndex,
                 &d_grid,
                 &d_gridUniqueIndex, &d_partUniqueIndex,
                 &d_particles,
                 getNumParticles(),
                 m_boundingBox, m_h,
                 accumForce,
                 dt);
*/

}

void NaiveGPUFluid::loadFluidVolumes() {

  FluidVolume h_volumes[m_volumes.size()];
  for (int i=0; i<m_volumes.size(); i++) {
    h_volumes[i] = m_volumes[i];
  }

 naive_initGPUFluid(&d_pos, &d_vel, &d_ppos, &d_dpos, &d_omega, 
                        &d_pcalc, &d_lambda, &d_grid, &d_gridCount, &d_gridInd, 
                        h_volumes, m_volumes.size(), m_boundingBox, m_h);

}

void NaiveGPUFluid::updateVBO(float* dptrvert) {
    naive_updateVBO(dptrvert, d_pos, getNumParticles());
}
#endif
