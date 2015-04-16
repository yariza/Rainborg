#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include "Fluid.h"
#include "FluidForce.h"


class Scene
{
public:
    Scene();
    ~Scene();

private:
    std::vector<Fluid> fluids; 
    //std::vector<Boundary> boundaries;
    std::vector<FluidForce*> fluid_forces;     
    
};

#endif
