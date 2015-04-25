#include "FluidRenderer.h"
#include "main.h"
#ifdef GPU_ENABLED
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gpu/GPUFluid.h"
//#include <helper_cuda.h>
//#include <helper_cuda_gl.h>

#endif

using namespace openglframework;

#define NUM_PARTS 100

FluidRenderer::FluidRenderer(Fluid* fluid)
: m_fluid(fluid)
, position_location(0)
, m_shader("shaders/point.vert", "shaders/point.frag")
{
    assert(m_fluid != NULL);

    if (g_gpu_mode) {
      #ifdef GPU_ENABLED
        vertices = new GLfloat[4 * NUM_PARTICLES];
        indices = new GLuint[NUM_PARTICLES];

        GLfloat x, y, z;
        for (int i=0; i<NUM_PARTICLES; i++) {
            x = static_cast <GLfloat> (rand()) / static_cast<GLfloat>(RAND_MAX/9.0);
            y = static_cast <GLfloat> (rand()) / static_cast<GLfloat>(RAND_MAX/9.0);
            z = static_cast <GLfloat> (rand()) / static_cast<GLfloat>(RAND_MAX/9.0);

            vertices[i*4 + 0] = x;
            vertices[i*4 + 1] = y;
            vertices[i*4 + 2] = z;
            vertices[i*4 + 3] = 1.0f;

            indices[i] = i;
        }

        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 4*NUM_PARTICLES*sizeof(GLfloat), vertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, NUM_PARTICLES*sizeof(GLuint), indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


        //cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );
        //cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
        cudaGLSetGLDevice(0);
        cudaGLRegisterBufferObject(vbo);
      #endif

    }
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
    glEnable(GL_POINT_SMOOTH);

    if (g_gpu_mode) {

        float *dptrvert=NULL;

        #ifdef GPU_ENABLED
        cudaGLMapBufferObject((void**)&dptrvert, vbo);
        updateVBOGPUFluid(dptrvert); // update content inside vbo object - implement this method in kernel!
        cudaGLUnmapBufferObject(vbo);

        glEnableVertexAttribArray(position_location);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(position_location, 4, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

        glDrawElements(GL_POINTS, NUM_PARTICLES, GL_UNSIGNED_INT, 0);

        glDisableVertexAttribArray(position_location);
        #endif
    }
    else {

        Vector3s* posArray = m_fluid->getFPPos();

        glBegin(GL_POINTS);
            for (int i=0; i<m_fluid->getNumParticles(); i++) {
                glVertex3f(posArray[i].x, posArray[i].y, posArray[i].z);
            }
        glEnd();

    }
    m_shader.unbind();
}
