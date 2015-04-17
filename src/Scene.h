#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include <iostream>
#include "Fluid.h"
#include "FluidForce.h"

//#define VERBOSE true

class Fluid;

class Scene
{
public:
    Scene();
    ~Scene();

    void insertFluid(const Fluid& newFluid); 
    void insertFluidForce(FluidForce* newFluidForce);
    std::vector<Fluid>& getFluids();
    std::vector<FluidForce*>& getFluidForces();

    private: 
    std::vector<Fluid> m_fluids; 
    //std::vector<Boundary> boundaries; 
    std::vector<FluidForce*> m_fluidForces;     
   

};

#endif
