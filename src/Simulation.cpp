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

void Simulation::load() {
    m_scene->load();
}

void Simulation::prepareForRender() {
    m_renderer->loadRenderers();
}

void Simulation::stepSystem(const scalar& dt) {

    m_stepper->stepScene(*m_scene, dt);
}

void Simulation::display(openglframework::GLFWViewer *viewer, int width, int height)
{
    m_renderer->render(viewer, width, height);
}
