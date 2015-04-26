#include "Fluid.h"

Fluid::Fluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors) 
: m_fpmass(mass)
, m_p0(p0)
, m_h(h)
, m_iters(iters)
, m_maxNeighbors(maxNeighbors)
, m_minNeighbors(minNeighbors)
, m_volumes()
{
}

Fluid::Fluid(const Fluid& otherFluid) {
    m_fpmass = otherFluid.m_fpmass;
    m_p0 = otherFluid.m_p0;
    m_h = otherFluid.m_h;
    m_iters = otherFluid.m_iters;
    m_maxNeighbors = otherFluid.m_maxNeighbors;
    m_minNeighbors = otherFluid.m_minNeighbors;
    m_volumes = otherFluid.m_volumes;
    m_boundingBox = otherFluid.m_boundingBox;
}

Fluid::~Fluid()
{
}

int Fluid::getNumParticles() const{
    int numParticles = 0;
    for (std::vector<FluidVolume>::size_type i=0; i<m_volumes.size(); i++) {

        numParticles += m_volumes[i].m_numParticles;
    }
    return numParticles;
}

void Fluid::setFPMass(scalar fpm){
    assert(fpm > 0); 
    m_fpmass = fpm;
}

scalar Fluid::getFPMass() const{
    return m_fpmass;
}

void Fluid::setRestDensity(scalar p0){
    m_p0 = p0;
}

scalar Fluid::getRestDensity() const{
    return m_p0;
}

void Fluid::setKernelH(scalar h){
    m_h = h; 
}

scalar Fluid::getKernelH() const{
    return m_h;
}

void Fluid::setNumIterations(int iter){
    m_iters = iter;
}

int Fluid::getNumIterations() const{
    return m_iters;
}

int Fluid::getMaxNeighbors() const{
    return m_maxNeighbors;
}

int Fluid::getMinNeighbors() const{
    return m_minNeighbors;
}

const std::vector<FluidVolume>& Fluid::getFluidVolumes() const {
    return m_volumes;
}

void Fluid::insertFluidVolume(FluidVolume& volume) {
    m_volumes.push_back(volume);
}

void Fluid::setBoundingBox(FluidBoundingBox& bound){
    m_boundingBox = bound;
}

const FluidBoundingBox& Fluid::getBoundingBox() const{
    return m_boundingBox;
}
