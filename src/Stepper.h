#ifndef __STEPPER_H__
#define __STEPPER_H__

#include "Scene.h"
#include "MathDefs.h"

class Stepper
{
public:
    Stepper();
    ~Stepper();
    bool stepScene(Scene& scene, scalar dt);
    
};

#endif
