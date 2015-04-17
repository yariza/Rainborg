#include "Fluid.h"
#include "Scene.h"

Fluid::Fluid(int numParticles) {
    // Allocate memory for m_pos, m_ppos, m_vel, m_accumForce? 

    m_numParticles = numParticles;
    m_pos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_ppos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_vel = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_accumForce = (scalar *)malloc(numParticles * 3 * sizeof(scalar));

    assert (m_pos != NULL);
    assert(m_ppos != NULL);
    assert(m_vel != NULL);
    assert(m_accumForce != NULL);
}

Fluid::~Fluid(){
    free(m_pos);
    free(m_ppos);
    free(m_vel);
    free(m_accumForce);
}

void Fluid::accumulateForce(Scene& scene){
    std::vector<FluidForce*> fluidForces = scene.getFluidForces();

    // init F to 0 
    memset (m_accumForce, 0, m_numParticles * 3 * sizeof(scalar)); 
    for(int i = 0; i < fluidForces.size(); ++i){
        fluidForces[i]->addGradEToTotal(m_pos, m_vel, m_fpmass, m_accumForce, m_numParticles);
    }
    
    // F *= -1.0/mass
    for(int i = 0; i < m_numParticles; ++i){
        m_accumForce[i] /= -m_fpmass; 
        m_accumForce[i+1] /= -m_fpmass;
        m_accumForce[i+2] /= -m_fpmass;
    }
}

void Fluid::updateVelocity(scalar dt){
    for(int i = 0; i < m_numParticles*3; ++i){
        m_vel[i] += m_accumForce[i] * dt; 
    }    
}

void Fluid::updatePredPosition(scalar dt){
    for(int i = 0; i < m_numParticles*3; ++i){
        m_ppos[i] = m_pos[i] + m_vel[i] * dt;
    }
}
