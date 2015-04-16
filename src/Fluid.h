#ifndef __FLUID_H__
#define __FLUID_H__

#include "MathDefs.h"

class Fluid{

public:
    
    

private: 
    int num_particles;
    scalar f_mass; // float particle mass, shared by all
    scalar f_p0; // rest density
    Vector3s *f_pos; // actual positions
    Vector3s *f_ppos; // predicted positions 
    Vector3s *f_vel; 
    
    // Add: Color? 
    // Boundary: Bounding Box           
};


#endif


