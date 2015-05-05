#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include <iostream>
#include "Fluid.h"
#include "FluidBoundary.h"
#include "FluidForce.h"

//#define VERBOSE true

class Fluid;

// The "scene" contains the fluids, the forces, and all of the boundary objects
class Scene
{
public:
    Scene();
    ~Scene();

    void insertFluid(Fluid* newFluid); 
    void insertFluidBoundary(FluidBoundary* newBoundary); 
    void insertFluidForce(FluidForce* newFluidForce);
    std::vector<Fluid*>& getFluids();
    std::vector<FluidBoundary*>& getFluidBoundaries(); 
    std::vector<FluidForce*>& getFluidForces();

    void load();

    private: 
    std::vector<Fluid*> m_fluids; 
    std::vector<FluidBoundary*> m_boundaries; 
    std::vector<FluidForce*> m_fluidForces;     

};

#endif
