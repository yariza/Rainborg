#include "Stepper.h"

Stepper::Stepper()
{}

Stepper::~Stepper()
{}

bool Stepper::stepScene(Scene& scene, scalar dt){

    // Our scene is static, but if it weren't, presumably apply forces to non-fluid objects here

    // Treat FluidForces and ForceForOtherThings separately... 

    // fluid.accumulateGradU() // makes more sense 
    

    // Accumulate forces for fluids
    // scene.accumulateFluidGradU(F); 
    // F *= -1.0 / mass
    // v += dt * F    
    // predpos += dt * v

    // Call fluid.step or equivalent





    return true;
}
