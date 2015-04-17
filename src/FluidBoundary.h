#ifndef __FLUID_BOUNDARY_H__
#define __FLUID_BOUNDARY_H__

#include "MathDefs.h"

class FluidBoundary {

public:
    virtual ~FluidBoundary();
    virtual void dealWithCollisions(scalar *pos, int numParticles) = 0; 

};


#endif
