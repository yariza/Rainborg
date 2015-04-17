#include "FluidSimpleGravityForce.h"


void FluidSimpleGravityForce::addGradEToTotal(scalar *f_pos, scalar *f_vel, scalar f_mass, scalar *f_accumGradU, int f_numParticles){
    
    // Wow this could be done in parallel too
    for(int i = 0; i < f_numParticles; ++i){
        f_accumGradU[i*3] -= f_mass * m_gravity[0];
        f_accumGradU[i*3+1] -= f_mass * m_gravity[1]; 
        f_accumGradU[i*3+2] -= f_mass * m_gravity[2];

    }
}


