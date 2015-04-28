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
  scalar sca2;
  scalar sca3;
  int id;
} grid_gpu_block_t;

extern "C" {
  void grid_initGPUFluid(Vector3s **g_pos, Vector3s **g_vel,
                         int **g_neighbors, int **g_gridIndex,
                         FluidVolume* h_volumes, int num_volumes,
                         FluidBoundingBox* h_boundingbox);

  void grid_updateVBO(float *vboptr, Vector3s *g_pos, int num_particles);
}
#endif
