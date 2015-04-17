#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>

#include "MathDefs.h"
//#include "Scene.h"
//#include "FluidForce.h"

class Scene;

class Fluid{

public:
 
    void accumulateGradU(Scene& scene);     
        

private: 
    int f_numParticles;
    scalar f_mass; // float particle mass, shared by all
    scalar f_p0; // rest density
    scalar *f_pos; // actual positinos
    scalar *f_ppos; // predicted positions
    scalar *f_vel; 

    // not much point reallocating memory for the same-sized force update vector each time; store here?
    scalar *f_accumGradU; 

    // Colors? 
    // Boundary?
 
};


#endif


