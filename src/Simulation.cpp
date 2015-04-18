#include "Simulation.h"

Simulation::Simulation(Scene* scene, Stepper* stepper, SceneRenderer* renderer)
: m_scene(scene)
, m_stepper(stepper)
, m_renderer(renderer)
{}

Simulation::~Simulation()
{
    if (m_scene) {
        delete m_scene;
        m_scene = NULL;
    }
    if (m_stepper) {
        delete m_stepper;
        m_stepper = NULL;
    }
    if (m_renderer) {
        delete m_renderer;
        m_renderer = NULL;
    }
}

