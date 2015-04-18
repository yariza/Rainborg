#include "FluidSimpleGravityForce.h"

FluidSimpleGravityForce::FluidSimpleGravityForce(const Vector3s& gravity)
: FluidForce()
, m_gravity(gravity) {
    assert(m_gravity == gravity);

if(VERBOSE)
    std::cout << "New fluid simple gravity force: " << m_gravity[0] << ", " << m_gravity[1] << ", " << m_gravity[2] << std::endl;
}

FluidSimpleGravityForce::FluidSimpleGravityForce(scalar gravX, scalar gravY, scalar gravZ)
: FluidForce()
, m_gravity(gravX, gravY, gravZ){
    assert(m_gravity[0] == gravX && m_gravity[1] == gravY && m_gravity[2] == gravZ);

if(VERBOSE)
    std::cout << "New fluid simple gravity force: " << m_gravity[0] << ", " << m_gravity[1] << ", " << m_gravity[2] << std::endl;
}

FluidSimpleGravityForce::~FluidSimpleGravityForce()
{}

void FluidSimpleGravityForce::addGradEToTotal(Vector3s *f_pos, Vector3s *f_vel, scalar f_mass, Vector3s *f_accumGradU, int f_numParticles){
    for(int i = 0; i < f_numParticles; ++i){
        f_accumGradU[i] -= f_mass * m_gravity; 
    }
}

//void FluidSimpleGravityForce::addGradEToTotal(scalar *f_pos, scalar *f_vel, scalar f_mass, scalar *f_accumGradU, int f_numParticles){
    
//    // Wow this could be done in parallel too
//    for(int i = 0; i < f_numParticles; ++i){
//        f_accumGradU[i*3] -= f_mass * m_gravity[0];
//        f_accumGradU[i*3+1] -= f_mass * m_gravity[1]; 
//        f_accumGradU[i*3+2] -= f_mass * m_gravity[2];
//    }
//}


