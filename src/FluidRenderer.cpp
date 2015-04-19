#include "FluidRenderer.h"

FluidRenderer::FluidRenderer(Fluid* fluid)
: m_fluid(fluid)
, m_shader("shaders/point.vert", "shaders/point.frag")
{
    assert(m_fluid != NULL);
}

FluidRenderer::~FluidRenderer() {
    m_fluid = NULL;
}

void FluidRenderer::render(GLFWViewer* viewer, int width, int height) {

    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_shader.bind();

    const Camera& camera = viewer->getCamera();

    Matrix4 matrixIdentity;
    matrixIdentity.setToIdentity();

    // m_shader.setVector3Uniform("cameraWorldPosition", viewer->getCamera().getOrigin());
    m_shader.setMatrix4x4Uniform("modelToWorldMatrix", matrixIdentity);
    m_shader.setMatrix4x4Uniform("worldToCameraMatrix", camera.getTransformMatrix().getInverse());
    m_shader.setMatrix4x4Uniform("projectionMatrix", camera.getProjectionMatrix());

    glPointSize(5.0);

    Vector3s* posArray = m_fluid->getFPPos();

    glEnable(GL_POINT_SMOOTH);
    glBegin(GL_POINTS);
        for (int i=0; i<m_fluid->getNumParticles(); i++) {
            glVertex3f(posArray[i].x, posArray[i].y, posArray[i].z);
        }
    glEnd();

    m_shader.unbind();
}
