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
, m_shader("shaders/point.vert", "shaders/point.frag")
{
    assert(m_fluid != NULL);

    if (g_gpu_mode) {
      #ifdef GPU_ENABLED
      int num_particles = fluid->getNumParticles();
      vertices = new GLfloat[8 * num_particles];
      indices = new GLuint[num_particles];

        std::cout << "gpu and render" << std::endl;

        GLfloat x, y, z;
        for (int i=0; i<fluid->getNumParticles(); i++) {
            indices[i] = i;
        }

        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        //glGenBuffers(1, &cbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 8*num_particles*sizeof(GLfloat), vertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_particles*sizeof(GLuint), indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
       
        /*
        glBindBuffer(GL_ARRAY_BUFFER, cbo);
        glBufferData(GL_ARRAY_BUFFER, num_particles*4*sizeof(GLfloat), fluid->getColors(), GL_STATIC_DRAW); 
        glBindBuffer(GL_ARRAY_BUFFER, 0);// okay?
        */

        //free(gl_cols);

        GPU_CHECKERROR(cudaGLRegisterBufferObject(vbo));
        
      #endif

    }
}

FluidRenderer::~FluidRenderer() {
    m_fluid = NULL;
}

void FluidRenderer::render(GLFWViewer* viewer, int width, int height) {

    std::cout << "started render" << std::endl;
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
    
        std::cout << "start cudaGLMapBufferObject" << std::endl;
    
        #ifdef GPU_ENABLED
        GPU_CHECKERROR(cudaGLMapBufferObject((void**)&dptrvert, vbo));
        std::cout << "before updateVBO" << std::endl;
        m_fluid->updateVBO(dptrvert);
        // updateVBOGPUFluid(dptrvert); // update conient inside vbo object - implement this method in kernel!
        std::cout << "unmap" << std::endl;
        GPU_CHECKERROR(cudaGLUnmapBufferObject(vbo));

//        glEnableVertexAttribArray(position_location);
//        std::cout << "enable vertex attrib array 1" << std::endl;
 //       glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        std::cout << "enable client states" << std::endl;
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
    
        //std::cout << "bind buffer" << std::endl;
        //glBindBuffer(GL_ARRAY_BUFFER, vbo);
        //std::cout << "vertex attrib pointer" << std::endl;
        //glVertexAttribPointer(position_location, 8, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexPointer(4, GL_FLOAT, 32, 0);
        glColorPointer(4, GL_FLOAT, 32, (void *)16);

        std::cout << "gl bind buffer" << std::endl; 
  //      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

        
       
        //glDrawElements(GL_POINTS, num_particles, GL_UNSIGNED_INT, 0);
        //glColor4f(0.0f, 1.0f, 0.0f, .6f);
    
        std::cout << "draw arrays" << std::endl;
        glDrawArrays(GL_POINTS, 0, num_particles);
        std::cout << "what now" << std::endl;

        //glDisableVertexAttribArray(position_location);
        //glDisableVertexAttribArray(1);
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
