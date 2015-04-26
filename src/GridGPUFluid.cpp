#include "GridGPUFluid.h"

GridGPUFluid::GridGPUFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors) 
: Fluid(mass, p0, h, iters, maxNeighbors, minNeighbors)
, m_eps(0.01)
{
}

GridGPUFluid::GridGPUFluid(GridGPUFluid& otherFluid)
: Fluid(otherFluid)
, m_eps(0.01)
{

}

GridGPUFluid::~GridGPUFluid() {

}

void GridGPUFluid::stepSystem(Scene& scene, scalar dt) {

}

void GridGPUFluid::loadFluidVolumes() {


}


