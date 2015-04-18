#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include "Scene.h"
#include "Stepper.h"
#include "SceneRenderer.h"

class Simulation
{
public:
    Simulation(Scene* scene, Stepper* stepper, SceneRenderer* renderer);
    ~Simulation();
    
private:
    Scene *m_scene;
    Stepper *m_stepper;
    SceneRenderer *m_renderer;
};

#endif
