#ifndef __OPENGL_RENDERER_H__
#define __OPENGL_RENDERER_H__

#include <openglframework.h>

class OpenGLRenderer
{
public:
    virtual void render(openglframework::GLFWViewer* viewer, int width, int height) = 0;

};

#endif