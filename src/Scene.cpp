#include "Scene.h"

Scene::Scene()
: m_fluids()
, m_fluidForces() 
{}

Scene::~Scene() {
    for(int i = 0; i < m_fluids.size(); ++i){
        if(m_fluids[i] != NULL){
            delete m_fluids[i]; 
            m_fluids[i] = NULL; 
        }
    }

    for(int i = 0; i < m_boundaries.size(); ++i){
        if(m_boundaries[i] != NULL){
            delete m_boundaries[i]; 
            m_boundaries[i] = NULL; 
        }
    }

    for(int i = 0; i < m_fluidForces.size(); ++i){
        if(m_fluidForces[i] != NULL){
            delete m_fluidForces[i];
            m_fluidForces[i] = NULL;
        }    
    }
}

std::vector<Fluid*>& Scene::getFluids(){
    return m_fluids;
}

std::vector<FluidBoundary*>& Scene::getFluidBoundaries(){
    return m_boundaries; 
}

std::vector<FluidForce*>& Scene::getFluidForces(){
    return m_fluidForces;
}

void Scene::insertFluid(Fluid* newFluid){
    m_fluids.push_back(newFluid);
} 

void Scene::insertFluidBoundary(FluidBoundary* newBound){
    m_boundaries.push_back(newBound); 
}

void Scene::insertFluidForce(FluidForce* newFluidForce){
    m_fluidForces.push_back(newFluidForce);
}


