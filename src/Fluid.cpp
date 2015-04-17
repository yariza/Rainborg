#include "Fluid.h"

void Fluid::accumulateGradU(Scene& scene){
    // init F to 0 
    memset (f_accumGradU, 0, f_numParticles * 3 * sizeof(scalar));

    // 


}

void Fluid::updateVelocity(scalar dt){
        // F *= -1.0/mass
        
        // v += dt * F
        // predpos += dt * v
        
 


}

void Fluid::updatePredPosition(scalar dt){

}
