#include "FluidBrick.h"

FluidBrick::FluidBrick(){
//    std::cout << "Empty FluidBrick construction" << std::endl;

}

FluidBrick::FluidBrick(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ, scalar eps) 
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

//    std::cout << "FluidBrick Construction from values" << std::endl;
}

FluidBrick::FluidBrick(const FluidBrick& otherBound){
    m_minX = otherBound.minX();
    m_maxX = otherBound.maxX();
    m_minY = otherBound.minY();
    m_maxY = otherBound.maxY();
    m_minZ = otherBound.minZ();
    m_maxZ = otherBound.maxZ();
    m_eps = otherBound.eps();

//    std::cout << "FluidBrick Copy Construction" << std::endl;
}


FluidBrick::~FluidBrick()
{}

FluidBrick& FluidBrick::operator=(const FluidBrick& otherBound){
    m_minX = otherBound.minX();
    m_minY = otherBound.minY(); 
    m_minZ = otherBound.minZ(); 
    m_maxX = otherBound.maxX();
    m_maxY = otherBound.maxY();
    m_maxZ = otherBound.maxZ();
    m_eps = otherBound.eps(); 

    return *this;
}

scalar FluidBrick::minX() const{
    return m_minX;
}

scalar FluidBrick::maxX() const{
    return m_maxX;
}

scalar FluidBrick::minY() const{
    return m_minY;
}

scalar FluidBrick::maxY() const{
    return m_maxY;
 }

scalar FluidBrick::minZ() const{
    return m_minZ;
}

scalar FluidBrick::maxZ() const{
    return m_maxZ;
}

scalar FluidBrick::eps() const{
    return m_eps;
}

scalar FluidBrick::width(){
    return m_maxX - m_minX; 
}

scalar FluidBrick::height(){
    return m_maxY - m_minY; 
}

scalar FluidBrick::depth(){
    return m_maxZ - m_minZ;
}

void FluidBrick::dealWithCollisions(Vector3s *pos, Vector3s *dpos, int numParticles){    
    scalar pposX; 
    scalar pposY; 
    scalar pposZ; 
    
    scalar midX = .5 * (m_minX + m_maxX); 
    scalar midY = .5 * (m_minY + m_maxY);
    scalar midZ = .5 * (m_minZ + m_maxZ);     


    for(int i = 0; i < numParticles; ++i){
        pposX = pos[i][0] + dpos[i][0];
        pposY = pos[i][1] + dpos[i][1];
        pposZ = pos[i][2] + dpos[i][2]; 

        if(pposX < m_minX - m_eps)
            return;
        if(pposX > m_maxX + m_eps)
            return;
        if(pposY < m_minY - m_eps)
            return;
        if(pposY > m_minY + m_eps)
            return;
        if(pposZ < m_minZ - m_eps)
            return;
        if(pposZ > m_maxZ + m_eps)
            return;

        // So inside the cube; push to nearest position?
        if(pposX > midX){
            dpos[i][0] = m_maxX + m_eps; 
        }
        else{
            dpos[i][0] = m_minX - m_eps; 
        }
        if(pposY > midY){
            dpos[i][1] = m_maxY + m_eps; 
        }
        else{
            dpos[i][1] = m_minY - m_eps; 
        }
        if(pposZ > midZ){
            dpos[i][2] = m_maxZ + m_eps; 
        }
        else{
            dpos[i][2] = m_minY - m_eps; 
        }

        
   }
}

