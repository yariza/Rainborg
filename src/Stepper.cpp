#include "Stepper.h"

Stepper::Stepper()
{}

Stepper::~Stepper()
{}

bool Stepper::stepScene(Scene& scene, scalar dt){

    // Our scene is static, but if it weren't, presumably apply forces to non-fluid objects here

    // Treat FluidForces and ForceForOtherThings separately... 

    // for all fluids in scene
    for(int i = 0; i < scene.fluids.size(); ++i){
        Fluid& fluid = scene.fluids[i];  

        fluid.accumulateGradU(scene); // makes more sense 
        fluid.updateVelocity(dt); 
        fluid.updatePredPosition(dt); 
       
    }




    return true;
}
