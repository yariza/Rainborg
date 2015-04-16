#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>

#include "MathDefs.h"
#include "FluidForce.h"

class Fluid{

public:
 
    void accumulateGradU(std::vector<FluidForce>& fluid_forces);     
    

private: 
    int f_numParticles;
    scalar f_mass; // float particle mass, shared by all
    scalar f_p0; // rest density
    Vector3s *f_pos; // actual positions
    Vector3s *f_ppos; // predicted positions 
    Vector3s *f_vel; 

    // Not much point recreating a 'force' update vector each time; store here?
    Vector3s *f_accumGradU;     
    // Add: Color? 
    // Boundary: Bounding Box           
};


#endif


