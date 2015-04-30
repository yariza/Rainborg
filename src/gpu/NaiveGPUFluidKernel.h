#ifndef __NAIVE_GPU_FLUID_KERNEL_H__
#define __NAIVE_GPU_FLUID_KERNEL_H__

#include "../FluidBoundingBox.h"
#include "../FluidVolume.h"
#include "../MathDefs.h"

#define naive_MIN_NEIGHBORS 3

#define naive_EPS .01
#define naive_VORTICITY 0
#define naive_VORT_EPS .0001
#define naive_XSPH 1
#define naive_C .0001
#define naive_ART_PRESSURE 1
#define naive_N 4
#define naive_DQ .3
#define naive_K .1

#define naive_COLOR_MODE_NORMAL 0
#define naive_COLOR_MODE_DEPTH 1
#define naive_COLOR_MODE 0

extern "C" {
 
    void naive_initGPUFluid(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd, char **d_color, int max_neigh,  
                        FluidVolume* h_volumes, int num_volumes,
                        FluidBoundingBox* h_boundingBox,
                        scalar h, bool random);

    void naive_updateVBO(float *vboptr, Vector3s *d_pos, char *d_color, int num_particles);

    void naive_cleanUp(Vector3s **d_pos, Vector3s **d_vel, Vector3s **d_ppos, Vector3s **d_dpos, Vector3s **d_omega, 
                        scalar **d_pcalc, scalar **d_lambda, int **d_grid, int **d_gridCount, int **d_gridInd, char **d_color);

    void naive_stepFluid(Vector3s *d_pos, Vector3s *d_vel, Vector3s *d_ppos, Vector3s *d_dpos, Vector3s* d_omega, scalar *d_pcalc, scalar *d_lambda, scalar fp_mass,
                      int num_particles, int max_neigh, int *d_grid, int *d_gridCount, int *d_gridInd, int iters, scalar p0,
                      FluidBoundingBox* h_boundingbox,
                      scalar h,
                      Vector3s accumForce,
                      scalar dt);

}
#endif
