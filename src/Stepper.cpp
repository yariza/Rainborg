#include "Stepper.h"

Stepper::Stepper()
{}

Stepper::~Stepper()
{}

bool Stepper::stepScene(Scene& scene, scalar dt){

    if(VERBOSE){
        std::cout << "Step Scene with dt: " << dt << std::endl;
    }

    // Our scene is static, but if it weren't, presumably apply forces to non-fluid objects here

    // Treat FluidForces and ForceForOtherThings separately... 

    // for all fluids in scene
    std::vector<Fluid*> fluids = scene.getFluids();
    for(int i = 0; i < fluids.size(); ++i){
        Fluid& fluid = *(fluids[i]);  
        
        fluid.stepSystem(scene, dt);

      
    }

    return true;
}
