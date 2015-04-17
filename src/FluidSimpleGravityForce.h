#ifndef __FLUID_SIMPLE_GRAVITY_FORCE_H__
#define __FLUID_SIMPLE_GRAVITY_FORCE_H__

#include "FluidForce.h"
#include "MathDefs.h"

class FluidSimpleGravityForce : public FluidForce {

public:
    FluidSimpleGravityForce(); 
    virtual ~FluidSimpleGravityForce();
    virtual void addGradEToTotal(scalar *f_pos, scalar *f_vel, scalar f_mass, scalar *f_accumGradU, int f_numParticles);


private: 
    Vector3s m_gravity; // Type? Vector3s sort of makes sense, especially since this will presumably be constant and just live somewhere
};

#endif
