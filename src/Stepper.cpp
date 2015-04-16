#include "Stepper.h"

Stepper::Stepper()
{}

Stepper::~Stepper()
{}

bool Stepper::stepScene(Scene& scene, scalar dt){

    // Accumulate forces 

    // scene.accumuluteGradU(F); 

    // Our scene is static, but if it weren't, presumably apply forces to non-fluid objects here

    // Call fluid.step or equivalent





    return true;
}
