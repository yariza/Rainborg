#ifndef __FLUID_BOUNDARY_H__
#define __FLUID_BOUNDARY_H__

#include "MathDefs.h"

class FluidBoundary {

public:
    virtual ~FluidBoundary();
    //virtual void dealWithCollisions(scalar *pos, scalar *dpos, int numParticles) = 0; 
    virtual void dealWithCollisions(Vector3s *pos, Vector3s *dpos, int numParticles) = 0; 


};


#endif
