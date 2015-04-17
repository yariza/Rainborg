#ifndef __FLUIDFORCE_H__
#define __FLUIDFORCE_H__

#include "MathDefs.h"

class FluidForce {

public:
    virtual ~FluidForce();
    virtual void addGradEToTotal(scalar *f_pos, scalar *f_vel, scalar f_mass, scalar *f_accumGradU, int f_numParticles) = 0;

};


#endif
