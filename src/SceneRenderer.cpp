#include "SceneRenderer.h"
#include "RenderingUtils.h"

SceneRenderer::SceneRenderer(Scene* scene)
: m_scene(scene)
, m_debug(false)
, m_renderers() {
    assert(m_scene != NULL);

    for(std::vector<Fluid*>::size_type i=0; i<m_scene->getFluids().size(); i++) {
        Fluid* fluid = m_scene->getFluids()[i];
        FluidRenderer* renderer = new FluidRenderer(fluid);
        m_renderers.push_back(renderer);
    }

}

SceneRenderer::~SceneRenderer() {
    for (std::vector<FluidRenderer*>::size_type i=0; i<m_renderers.size(); i++) {
        if (m_renderers[i] != NULL) {
            delete m_renderers[i];
            m_renderers[i] = NULL;
        }
    }
}

void SceneRenderer::toggleDebugMode() {
    m_debug = !m_debug;
}

void SceneRenderer::render(GLFWViewer* viewer, int width, int height) {

    float radius = 10.0f;
    Vector3 center(0.0, 0.0, 0.0);

    viewer->setScenePosition(center, radius);

    for (std::vector<FluidRenderer*>::size_type i=0; i<m_renderers.size(); i++) {
        m_renderers[i]->render(viewer, width, height);
    }
}
