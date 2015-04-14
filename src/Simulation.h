#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include "Scene.h"
#include "Stepper.h"
#include "Renderer.h"

class Simulation
{
public:
    Simulation();
    ~Simulation();
    
private:
    Scene *m_scene;
    Stepper *m_stepper;
    Renderer *m_renderer;
};

#endif
