#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include "Fluid.h"
#include "FluidForce.h"

class Fluid;

class Scene
{
public:
    Scene();
    ~Scene();

    std::vector<Fluid> fluids; 
    //std::vector<Boundary> boundaries;
    std::vector<FluidForce*> fluid_forces;     
   
private: 
// be a terrible person and make everything public
 
};

#endif
