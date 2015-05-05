#ifndef __FLUID_SIMPLE_GRAVITY_FORCE_H__
#define __FLUID_SIMPLE_GRAVITY_FORCE_H__

#include <iostream>
#include "FluidForce.h"
#include "MathDefs.h"

#define VERBOSE false

// Simple gravity! 
class FluidSimpleGravityForce : public FluidForce {

public:
    FluidSimpleGravityForce(const Vector3s& gravity);
    FluidSimpleGravityForce(scalar gravX, scalar gravY, scalar gravZ); 


    virtual ~FluidSimpleGravityForce();
    virtual void addGradEToTotal(Vector3s *f_pos, Vector3s *f_vel, scalar f_mass, Vector3s *f_accumGradU, int f_numParticles);

    virtual Vector3s getGlobalForce();

private: 
    Vector3s m_gravity;
};

#endif
