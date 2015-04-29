#ifndef __NAIVE_GPU_FLUID_KERNEL_H__
#define __NAIVE_GPU_FLUID_KERNEL_H__

#include "../FluidBoundingBox.h"
#include "../FluidVolume.h"
#include "../MathDefs.h"

#define VORTICITY 0
#define XSPH 0
#define ART_PRESSURE 0

extern "C" {
 
    void naive_initGPUFluid(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd, 
                        FluidVolume* h_volumes, int num_volumes,
                        FluidBoundingBox* h_boundingBox,
                        scalar h);

    void naive_updateVBO(float *vboptr, Vector3s *d_pos, int num_particles);

    void naive_cleanUp(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd);

    void naive_stepFluid(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, Vector3s *d_dpos, scalar fp_mass,
                      int num_particles,
                      FluidBoundingBox* h_boundingbox,
                      scalar h,
                      Vector3s accumForce,
                      scalar dt);
   /*
  void naive_initGPUFluid(int **g_neighbors, int **g_gridIndex,
                         int **g_grid,
                         int **g_gridUniqueIndex, int **g_partUniqueIndex,
                         grid_gpu_block_t **g_particles,
                         FluidVolume* h_volumes, int num_volumes,
                         FluidBoundingBox* h_boundingbox,
                         scalar h);

  void naive_stepFluid(int **g_neighbors, int **g_gridIndex,
                      int **g_grid,
                      int **g_gridUniqueIndex, int **g_partUniqueIndex,
                      grid_gpu_block_t **g_particles,
                      int num_particles,
                      FluidBoundingBox* h_boundingbox,
                      scalar h,
                      Vector3s accumForce,
                      scalar dt);

  void naive_updateVBO(float *vboptr, grid_gpu_block_t *g_particles, int num_particles);
*/
}
#endif
