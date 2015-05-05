#ifndef __FLUID_BOUNDARY_H__
#define __FLUID_BOUNDARY_H__

#include "MathDefs.h"

// Abstract class for things the fluid can interact with
class FluidBoundary {

public:
    virtual ~FluidBoundary();
    virtual void dealWithCollisions(Vector3s *pos, Vector3s *dpos, int numParticles) = 0; 


};


#endif
