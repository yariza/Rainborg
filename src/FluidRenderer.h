#ifndef __FLUID_RENDERER_H__
#define __FLUID_RENDERER_H__

#include "SerialFluid.h"
#include "OpenGLRenderer.h"

class FluidRenderer : OpenGLRenderer
{
public:
    FluidRenderer(Fluid* fluid);
    ~FluidRenderer();

    virtual void render(openglframework::GLFWViewer* viewer, int width, int height);
protected:
    Fluid* m_fluid;

private:
    openglframework::Shader m_shader;

    //gpu stuff
    GLuint vbo;
    GLuint ibo;

    GLfloat *vertices;
    GLuint *indices;

    GLuint position_location;
};

#endif