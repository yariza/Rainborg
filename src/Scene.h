#ifndef __SCENE_H__
#define __SCENE_H__

#include <vector>
#include "Fluid.h"
#include "Force.h"


class Scene
{
public:
    Scene();
    ~Scene();

private:
    std::vector<Fluid> fluids; 
    //std::vector<Boundary> boundaries;
    std::vector<Force> forces;     
    
};

#endif
