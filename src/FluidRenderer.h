#ifndef __FLUID_RENDERER_H__
#define __FLUID_RENDERER_H__

#include "Fluid.h"
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
};

#endif