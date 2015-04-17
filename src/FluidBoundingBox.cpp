#include "FluidBoundingBox.h"

FluidBoundingBox::FluidBoundingBox(){
//    std::cout << "Empty FluidBoundingBox construction" << std::endl;

}

FluidBoundingBox::FluidBoundingBox(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ, scalar eps) 
: FluidBoundary()
, m_minX(minX)
, m_maxX(maxX)
, m_minY(minY)
, m_maxY(maxY)
, m_minZ(minZ)
, m_maxZ(maxZ) 
, m_eps(eps) {

    assert(minX < maxX);
    assert(minY < maxY);
    assert(minZ < maxZ);
    assert(eps > 0); 

//    std::cout << "FluidBoundingBox Construction from values" << std::endl;
}

FluidBoundingBox::FluidBoundingBox(const FluidBoundingBox& otherBound){
    m_minX = otherBound.minX();
    m_maxX = otherBound.maxX();
    m_minY = otherBound.minY();
    m_maxY = otherBound.maxY();
    m_minZ = otherBound.minZ();
    m_maxZ = otherBound.maxZ();
    m_eps = otherBound.eps();

//    std::cout << "FluidBoundingBox Copy Construction" << std::endl;
}


FluidBoundingBox::~FluidBoundingBox()
{}

FluidBoundingBox& FluidBoundingBox::operator=(const FluidBoundingBox& otherBound){
    m_minX = otherBound.minX();
    m_minY = otherBound.minY(); 
    m_minZ = otherBound.minZ(); 
    m_maxX = otherBound.maxX();
    m_maxY = otherBound.maxY();
    m_maxZ = otherBound.maxZ();

    return *this;
}

scalar FluidBoundingBox::minX() const{
    return m_minX;
}

scalar FluidBoundingBox::maxX() const{
    return m_maxX;
}

scalar FluidBoundingBox::minY() const{
    return m_minY;
}

scalar FluidBoundingBox::maxY() const{
    return m_maxY;
 }

scalar FluidBoundingBox::minZ() const{
    return m_minZ;
}

scalar FluidBoundingBox::maxZ() const{
    return m_maxZ;
}

scalar FluidBoundingBox::eps() const{
    return m_eps;
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

// So... I guess this particular function deals with all the particles at once? 
void FluidBoundingBox::dealWithCollisions(scalar *pos, int numParticles){    
    for(int i = 0; i < numParticles; ++i){
        if(pos[i*3] < m_minX + m_eps){
            pos[i*3] = m_minX + m_eps; 
        }       
        else if(pos[i*3] > m_maxX - m_eps){
            pos[i*3] = m_maxX - m_eps;
        }
        if(pos[i*3+1] < m_minY + m_eps){
            pos[i*3+1] = m_minY + m_eps; 
        }       
        else if(pos[i*3+1] > m_maxY - m_eps){
            pos[i*3+1] = m_maxY - m_eps;
        }
        if(pos[i*3+2] < m_minZ + m_eps){
            pos[i*3+2] = m_minZ + m_eps; 
        }       
        else if(pos[i*3+2] > m_maxZ - m_eps){
            pos[i*3+2] = m_maxZ - m_eps;
        }
    }
}

