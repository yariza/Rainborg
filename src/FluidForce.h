#ifndef __FLUIDFORCE_H__
#define __FLUIDFORCE_H__

#include "MathDefs.h"

class FluidForce {

public:
    virtual ~FluidForce();
    virtual void addGradEToTotal(Vector3s *f_pos, Vector3s *f_vel, scalar f_mass, Vector3s *f_accumGradU, int f_numParticles) = 0;


};


#endif
