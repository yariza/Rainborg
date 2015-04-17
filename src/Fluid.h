#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>
#include <cstring>
#include <iostream>

#include "MathDefs.h"
#include "FluidBoundingBox.h"

class Scene;

class Fluid{

public:

    Fluid(int numParticles);
    Fluid(const Fluid& otherFluid);
    ~Fluid();

    void stepSystem(Scene& scene, scalar dt);
    void setFPMass(scalar fpm); 
    void setRestDensity(scalar p0);
    void setFPPos(int fp, const Vector3s& pos);
    void setFPVel(int fp, const Vector3s& vel);
    void setKernelH(scalar h);
    void setBoundingBox(FluidBoundingBox& newBound);

    int getNumParticles() const;
    scalar getFPMass() const;
    scalar getRestDensity() const;
    scalar getKernelH() const;
    scalar* getFPPos() const;
    scalar* getFPVel() const;
    const FluidBoundingBox& getBoundingBox() const;
    
private: 

    void accumulateForce(Scene& scene);     
    void updateVelocity(scalar dt); 
    void updatePredPosition(scalar dt); 
    //void findNeighbors(); 
    //void dealWithCollisions(Scene& scene); // ... Deal with scene collisions
    //void preserveOwnBoundary(); // Make sure within own bounding box
    
    
    int m_numParticles;
    scalar m_fpmass; // float particle mass, shared by all
    scalar m_p0; // rest density
    scalar *m_pos; // actual positinos
    scalar *m_ppos; // predicted positions
    scalar *m_vel; 

    scalar m_h; // kernel radius

    scalar *m_accumForce; 

    // Neighbors? 
    // lambdas?
    // dP? 

    FluidBoundingBox m_boundingBox; 

    // Colors? 
    // Boundary?
 
};


#endif


