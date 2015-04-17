#ifndef __FLUID_BOUNDING_BOX_H__
#define __FLUID_BOUNDING_BOX_H__

#include <iostream>
#include "Boundary.h"
#include "MathDefs.h"

class FluidBoundingBox : public Boundary {
public: 
    FluidBoundingBox(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ);
    virtual ~FluidBoundingBox(); 
    virtual void dealWithCollisions(scalar *pos); 

    // Need the bounds for calculating the grid
    scalar minX(); 
    scalar maxX(); 
    scalar minY();
    scalar maxY(); 
    scalar minZ();
    scalar maxZ();
    
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
};


#endif
