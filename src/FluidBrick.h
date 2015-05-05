#ifndef __FLUID_BRICK_H__
#define __FLUID_BRICK_H__

#include "FluidBoundary.h"
#include "MathDefs.h"

// Unused at the moment
// A volume that keeps the fluid OUTSIDE its bounds (as opposed to FluidBoundingBox, which keeps it in)
class FluidBrick : public FluidBoundary {
public: 
    FluidBrick();
    FluidBrick(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ, scalar eps = .01);
    FluidBrick(const FluidBrick& otherBrick); 
    FluidBrick& operator=(const FluidBrick& otherBrick);

    virtual ~FluidBrick(); 
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
