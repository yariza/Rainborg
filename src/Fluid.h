#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>
#include <cstring>
#include <iostream>

#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"

class Scene;

class Fluid{

public:

    Fluid(int numParticles, scalar mass, scalar p0, scalar h, int iters, int maxNeigh = 20, int minNeighbor = 3);
    Fluid(const Fluid& otherFluid);
    ~Fluid();

    void stepSystem(Scene& scene, scalar dt);
    void setFPMass(scalar fpm); 
    void setRestDensity(scalar p0);
    void setFPPos(int fp, const Vector3s& pos);
    void setFPVel(int fp, const Vector3s& vel);
    void setKernelH(scalar h);
    void setNumIterations(int iter); 
    void setBoundingBox(FluidBoundingBox& newBound);
    void setColor(int i, const Vector4s& col); 
    void insertFluidVolume(FluidVolume& volume);

    int getNumParticles() const;
    int getNumIterations() const;
    int getMaxNeighbors() const;
    int getMinNeighbors() const; 
    scalar getFPMass() const;
    scalar getRestDensity() const;
    scalar getKernelH() const;
    //scalar* getFPPos() const;
    //scalar* getFPVel() const;
    Vector3s* getFPPos() const;
    Vector3s* getFPVel() const; 
    Vector4s* getColors() const;
    const FluidBoundingBox& getBoundingBox() const;
    const std::vector<FluidVolume>& getFluidVolumes() const;
    
private: 

    void accumulateForce(Scene& scene);     
    void updateVelocityFromForce(scalar dt); 
    void updatePredPosition(scalar dt); 
    void clearGrid();
    void buildGrid(); // how to parallelize? 
//    void getGridIdx(scalar x, scalar y, scalar z, int& idx); // make separate in case of smarter coallesced memory access    
    void getGridIdx(Vector3s &pos, int& idx, int& idy, int &idz);  
    int getGridIdx(int i, int j, int k); 

    void calculatePressures(); 
    void calculateLambdas(); 
    void calculatedPos(); 
  
    void dealWithCollisions(Scene& scene); // ... Deal with scene collisions
    void preserveOwnBoundary(); // Make sure within own bounding box

    Vector3s calcGradConstraint(Vector3s& pi, Vector3s& pj);
    Vector3s calcGradConstraintAtI(int p);  

    void applydPToPredPos(); 
    void recalculateVelocity(scalar dt); 
    // Vorticity confinement, XSPH
    void updateFinalPosition();     
    
    
    int m_numParticles;
    int m_maxNeighbors; 
    int m_minNeighbors; // minimum number of neighbors to do calculations

    int *m_grid; // grid of particles, x, then y, then z, size w*h*d*maxNeighbors
    int *m_gridCount; // store number of particles in grid so far (needed for GPU)
    int *m_gridInd; // indices for grid per particle
    int m_gridX; 
    int m_gridY; 
    int m_gridZ; 

    scalar m_fpmass; // float particle mass, shared by all
    scalar m_p0; // rest pressure
    scalar *m_pcalc; // calculated pressure
    //scalar *m_pos; // actual positinos
    //scalar *m_ppos; // predicted positions
    scalar *m_lambda; // calculated lambdas 
    //scalar *m_dpos; // change in positions
    //scalar *m_vel; 

    Vector3s *m_pos; 
    Vector3s *m_ppos; 
    Vector3s *m_dpos; 
    Vector3s *m_vel; 

    scalar m_eps; 
    scalar m_h; // kernel radius
    int m_iters; // how many iterations through solver?

    //scalar *m_accumForce; 
    Vector3s *m_accumForce;

    FluidBoundingBox m_boundingBox; 

    // Colors? 
    Vector4s *m_colors;     

    std::vector<FluidVolume> m_volumes;
 
};


#endif


