#ifndef __SERIAL_FLUID_H__
#define __SERIAL_FLUID_H__

#include <vector>
#include <cstring>
#include <iostream>

#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"
#include "Fluid.h"

class Scene;

// The serial implementation of a fluid
class SerialFluid : public Fluid{

public:

    SerialFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeigh = 20, int minNeighbor = 3);
    SerialFluid(const SerialFluid& otherFluid);
    virtual ~SerialFluid();
    virtual void stepSystem(Scene& scene, scalar dt);
    virtual void setFPVel(int fp, const Vector3s& vel);
    virtual void setBoundingBox(FluidBoundingBox* newBound);
    virtual void setColor(int i, const Vector4s& col); 

    virtual void loadFluidVolumes();
	
    virtual void updateVBO(float* vboptr);

    virtual Vector3s* getFPPos() const;
    virtual Vector3s* getFPVel() const; 

private: 

    void accumulateForce(Scene& scene);     
    void updateVelocityFromForce(scalar dt); 
    void updatePredPosition(scalar dt); 
    void clearGrid();
    void buildGrid(); // how to parallelize? 
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
    
    

    int *m_grid; // grid of particles, x, then y, then z, size w*h*d*maxNeighbors
    int *m_gridCount; // store number of particles in grid so far (needed for GPU)
    int *m_gridInd; // indices for grid per particle
    int m_gridX; // width
    int m_gridY; // height
    int m_gridZ; // depth

    scalar *m_pcalc; // calculated pressure
    scalar *m_lambda; // calculated lambdas 

    Vector3s *m_pos; // particle positions
    Vector3s *m_ppos; // particle predicted positions
    Vector3s *m_dpos; // particle change in position
    Vector3s *m_vel; // particle velocities

    scalar m_eps; 

    Vector3s *m_accumForce;


 
};


#endif


