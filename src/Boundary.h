#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include "MathDefs.h"

class Boundary {

public:
    virtual ~Boundary();
    virtual void dealWithCollisions(scalar *pos) = 0; 

};


#endif
