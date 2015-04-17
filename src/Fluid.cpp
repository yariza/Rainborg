#include "Fluid.h"
#include "Scene.h"

void Fluid::accumulateForce(Scene& scene){
    // init F to 0 
    memset (f_accumForce, 0, f_numParticles * 3 * sizeof(scalar));
    
    for(int i = 0; i < scene.fluidForces.size(); ++i){
        scene.fluidForces[i]->addGradEToTotal(f_pos, f_vel, f_mass, f_accumForce, f_numParticles);
    }
    
    // F *= -1.0/mass
    for(int i = 0; i < f_numParticles; ++i){
        f_accumForce[i] /= -f_mass; 
        f_accumForce[i+1] /= -f_mass;
        f_accumForce[i+2] /= -f_mass;
    }
}

void Fluid::updateVelocity(scalar dt){
    for(int i = 0; i < f_numParticles*3; ++i){
        f_vel[i] += f_accumForce[i] * dt; 
    }    
}

void Fluid::updatePredPosition(scalar dt){
    for(int i = 0; i < f_numParticles*3; ++i){
        f_ppos[i] = f_pos[i] + f_vel[i] * dt;
    }
}
