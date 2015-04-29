#ifndef __NAIVE_GPU_FLUID_KERNEL_H__
#define __NAIVE_GPU_FLUID_KERNEL_H__

#include "../FluidBoundingBox.h"
#include "../FluidVolume.h"
#include "../MathDefs.h"

#define naive_MIN_NEIGHBORS 3

#define naive_EPS .01
#define naive_VORTICITY 0
#define naive_VORT_EPS .0001
#define naive_XSPH 0
#define naive_C .0001
#define naive_ART_PRESSURE 0
#define naive_N 4
#define naive_DQ .3
#define naive_K .1


extern "C" {
 
    void naive_initGPUFluid(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd, int max_neigh,  
                        FluidVolume* h_volumes, int num_volumes,
                        FluidBoundingBox* h_boundingBox,
                        scalar h);

    void naive_updateVBO(float *vboptr, Vector3s *d_pos, int num_particles);

    void naive_cleanUp(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd);

    void naive_stepFluid(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, Vector3s *d_dpos, Vector3s* d_omega, scalar *d_pcalc, scalar *d_lambda, scalar fp_mass,
                      int num_particles, int max_neigh, int *d_grid, int *d_gridCount, int *d_gridInd, int iters, scalar p0,
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
