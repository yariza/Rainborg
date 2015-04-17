#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>
#include <cstring>
#include <iostream>

#include "MathDefs.h"
//#include "Scene.h"
//#include "FluidForce.h"

class Scene;

class Fluid{

public:

    Fluid(int numParticles);
    Fluid(const Fluid& otherFluid);
    ~Fluid();
 
    void accumulateForce(Scene& scene);     
    void updateVelocity(scalar dt); 
    void updatePredPosition(scalar dt); 
    void setFPMass(scalar fpm); 
    void setRestDensity(scalar p0);
    void setFPPos(int fp, const Vector3s& pos);
    void setFPVel(int fp, const Vector3s& vel);

    int getNumParticles() const;
    scalar getFPMass() const;
    scalar getRestDensity() const;
    scalar* getFPPos() const;
    scalar* getFPVel() const;
    
private: 
    
    int m_numParticles;
    scalar m_fpmass; // float particle mass, shared by all
    scalar m_p0; // rest density
    scalar *m_pos; // actual positinos
    scalar *m_ppos; // predicted positions
    scalar *m_vel; 

    scalar *m_accumForce; 

    // Colors? 
    // Boundary?
 
};


#endif


