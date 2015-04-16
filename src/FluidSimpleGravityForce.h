#ifndef __FLUID_SIMPLE_GRAVITY_FORCE_H__
#define __FLUID_SIMPLE_GRAVITY_FORCE_H__

#include "FluidForce.h"
#include "MathDefs.h"

class FluidSimpleGravityForce : public FluidForce {

public:
    FluidSimpleGravityForce(); 
    virtual ~FluidSimpleGravityForce();
    virtual void addGradEToTotal(Vector3s *f_pos, Vector3s *f_vel, scalar f_mass, Vector3s *f_accumGradU, int f_numParticles);


private: 
    Vector3s m_gravity; 
};

#endif
