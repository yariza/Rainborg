#include "Scene.h"

Scene::Scene()
: m_fluids()
, m_fluidForces() 
{}

Scene::~Scene() {
    for(int i = 0; i < m_fluidForces.size(); ++i){
        if(m_fluidForces[i] != NULL){
            delete m_fluidForces[i];
            m_fluidForces[i] = NULL;
        }    
    }
}

std::vector<Fluid>& Scene::getFluids(){
    return m_fluids;
}

std::vector<FluidForce*>& Scene::getFluidForces(){
    return m_fluidForces;
}

void Scene::insertFluid(const Fluid& newFluid){
    m_fluids.push_back(newFluid);
} 


void Scene::insertFluidForce(FluidForce* newFluidForce){
    m_fluidForces.push_back(newFluidForce);
}


