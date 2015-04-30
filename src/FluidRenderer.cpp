#include "FluidRenderer.h"
#include "main.h"
#ifdef GPU_ENABLED
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda.h>
//#include "gpu/GPUFluidNaive.h"
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#endif

using namespace openglframework;

#define NUM_PARTS 100

#ifdef GPU_ENABLED

#define GPU_CHECKERROR(err) (gpuCheckError(err, __FILE__, __LINE__))
static void gpuCheckError(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    }   
}

#endif


FluidRenderer::FluidRenderer(Fluid* fluid)
: m_fluid(fluid)
, position_location(0)
, color_location(12)
, m_shader("shaders/point.vert", "shaders/point.frag")
{
    assert(m_fluid != NULL);

    if (g_gpu_mode) {
      #ifdef GPU_ENABLED
      int num_particles = fluid->getNumParticles();
      vertices = new GLfloat[4 * num_particles];
      indices = new GLuint[num_particles];

        GLfloat x, y, z;
        for (int i=0; i<fluid->getNumParticles(); i++) {
            indices[i] = i;
        }

        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 4*num_particles*sizeof(GLfloat), vertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_particles*sizeof(GLuint), indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        GPU_CHECKERROR(cudaGLRegisterBufferObject(vbo));
        
      #endif

    }
}

FluidRenderer::~FluidRenderer() {
    m_fluid = NULL;
}

void FluidRenderer::render(GLFWViewer* viewer, int width, int height) {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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

    //glEnable(GL_VERTEX_ARRAY);
    //glEnable(GL_COLOR_ARRAY);

    int num_particles = m_fluid->getNumParticles();
 
    if (g_gpu_mode) {

        float *dptrvert=NULL;
    
    
        #ifdef GPU_ENABLED
        GPU_CHECKERROR(cudaGLMapBufferObject((void**)&dptrvert, vbo));
        m_fluid->updateVBO(dptrvert);
        GPU_CHECKERROR(cudaGLUnmapBufferObject(vbo));

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
    
        glVertexPointer(3, GL_FLOAT, 16, (void *)position_location);
        glColorPointer(4, GL_UNSIGNED_BYTE, 16, (void *)color_location);

        glDrawArrays(GL_POINTS, 0, num_particles);

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);

        #endif
    }
    else {
        SerialFluid* serialFluid = dynamic_cast<SerialFluid*>(m_fluid);
        if (!serialFluid) {
          std::cerr << "FluidRenderer: Fluid type is not Serial!" << std::endl;
          exit(1);
        }

        Vector3s* posArray = serialFluid->getFPPos();
        Vector4s* colorArray = serialFluid->getColors();

        
        glBegin(GL_POINTS);
            for (int i=0; i<serialFluid->getNumParticles(); i++) {
                //glColor3f(); 
                glColor4f(colorArray[i][0], colorArray[i][1], colorArray[i][2], colorArray[i][3]);
                //glColor3f(colorArray[i][0], colorArray[i][1], colorArray[i][2]);
                glVertex3f(posArray[i].x, posArray[i].y, posArray[i].z);
            }
        glEnd();
        
    }
    m_shader.unbind();
}
