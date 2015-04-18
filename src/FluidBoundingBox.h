#ifndef __FLUID_BOUNDING_BOX_H__
#define __FLUID_BOUNDING_BOX_H__

#include <iostream>
#include "FluidBoundary.h"
#include "MathDefs.h"

class FluidBoundingBox : public FluidBoundary {
public: 
    FluidBoundingBox();
    FluidBoundingBox(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ, scalar eps = .01);
    FluidBoundingBox(const FluidBoundingBox& otherBound); 
    FluidBoundingBox& operator=(const FluidBoundingBox& otherBound);

    virtual ~FluidBoundingBox(); 
    //virtual void dealWithCollisions(scalar *pos, scalar *dpos, int numParticles); 
    virtual void dealWithCollisions(Vector3s *pos, Vector3s *dpos, int numParticles); 

    // Need the bounds for calculating the grid
    scalar minX() const; 
    scalar maxX() const; 
    scalar minY() const;
    scalar maxY() const; 
    scalar minZ() const;
    scalar maxZ() const;
    scalar eps() const;
    
    scalar width(); // X
    scalar height(); // Y
    scalar depth(); // Z

private: 
    scalar m_minX; 
    scalar m_maxX;
    scalar m_minY;
    scalar m_maxY;
    scalar m_minZ;
    scalar m_maxZ;
    scalar m_eps; // move things inside bounding box by eps
};


#endif
