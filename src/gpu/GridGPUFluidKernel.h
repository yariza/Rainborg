#ifndef __GRID_GPU_FLUID_KERNEL_H__
#define __GRID_GPU_FLUID_KERNEL_H__

#include "../FluidBoundingBox.h"
#include "../FluidVolume.h"
#include "../MathDefs.h"

typedef struct {
  Vector3s pos;
  Vector3s vec1;
  Vector3s vec2;
  Vector3s vec3;
  scalar sca1;
  char r;
  char g;
  char b;
  char a;
  char num_neighbors;
} grid_gpu_block_t;

extern "C" {
  void grid_initGPUFluid(int **g_neighbors, int **g_gridIndex,
                         int **g_grid,
                         grid_gpu_block_t **g_particles,
                         FluidVolume* h_volumes, int num_volumes,
                         FluidBoundingBox* h_boundingbox,
                         scalar h);

  void grid_stepFluid(int **g_neighbors, int **g_gridIndex,
                      int **g_grid,
                      grid_gpu_block_t **g_particles,
                      int num_particles,
                      FluidBoundingBox* h_boundingbox,
                      int iters,
                      scalar mass,
                      scalar h,
                      scalar p0,
                      Vector3s accumForce,
                      scalar dt);

  void grid_updateVBO(float *vboptr, grid_gpu_block_t *g_particles, int num_particles);
}
#endif
