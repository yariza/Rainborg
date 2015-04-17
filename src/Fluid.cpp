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


Fluid::Fluid(const Fluid& otherFluid){
    m_numParticles = otherFluid.getNumParticles();
    m_fpmass = otherFluid.getFPMass();
    m_p0 = otherFluid.getRestDensity();
    
    // Allocate memory 
    m_pos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_ppos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_vel = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_accumForce = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));

    assert (m_pos != NULL);
    assert(m_ppos != NULL);
    assert(m_vel != NULL);
    assert(m_accumForce != NULL);

    // Set positions, velocity 
    // Note: predicted positions, accumulatedForces are recalculated each time step so no point copying those

    memcpy(m_pos, otherFluid.getFPPos(), m_numParticles * 3 * sizeof(scalar));
    memcpy(m_vel, otherFluid.getFPVel(), m_numParticles * 3 * sizeof(scalar));

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

void Fluid::setFPMass(scalar fpm){
    assert(fpm > 0); 
    m_fpmass = fpm;
}

void Fluid::setRestDensity(scalar p0){
    m_p0 = p0;
}

void Fluid::setFPPos(int fp, const Vector3s& pos){
    assert(fp >= 0 && fp < m_numParticles);
    
    m_pos[fp*3] = pos[0];
    m_pos[fp*3+1] = pos[1];
    m_pos[fp*3+2] = pos[2];
}

void Fluid::setFPVel(int fp, const Vector3s& vel){
    assert(fp >= 0 && fp < m_numParticles);
    
    m_vel[fp*3] = vel[0];
    m_vel[fp*3+1] = vel[1];
    m_vel[fp*3+2] = vel[2];
}

int Fluid::getNumParticles() const{
    return m_numParticles;
}

scalar Fluid::getFPMass() const{
    return m_fpmass;
}

scalar Fluid::getRestDensity() const{
    return m_p0;
}

scalar* Fluid::getFPPos() const{
    return m_pos;
}

scalar* Fluid::getFPVel() const{
    return m_vel;
}
