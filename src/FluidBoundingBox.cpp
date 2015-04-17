#include "FluidBoundingBox.h"

FluidBoundingBox::FluidBoundingBox(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ)
: Boundary()
, m_minX(minX)
, m_maxX(maxX)
, m_minY(minY)
, m_maxY(maxY)
, m_minZ(minZ)
, m_maxZ(maxZ) {

    assert(minX < maxX);
    assert(minY < maxY);
    assert(minZ < maxZ);
}


FluidBoundingBox::~FluidBoundingBox()
{}

scalar FluidBoundingBox::minX(){
    return m_minX;
}

scalar FluidBoundingBox::maxX(){
    return m_maxX;
}

scalar FluidBoundingBox::minY(){
    return m_minY;
}

scalar FluidBoundingBox::maxY(){
    return m_maxY;
}

scalar FluidBoundingBox::minZ(){
    return m_minZ;
}

scalar FluidBoundingBox::maxZ(){
    return m_maxZ;
}

scalar FluidBoundingBox::width(){
    return m_maxX - m_minX; 
}

scalar FluidBoundingBox::height(){
    return m_maxY - m_minY; 
}

scalar FluidBoundingBox::depth(){
    return m_maxZ - m_minZ;
}

void FluidBoundingBox::dealWithCollisions(scalar *pos){


}

