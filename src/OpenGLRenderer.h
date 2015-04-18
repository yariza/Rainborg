#ifndef __OPENGL_RENDERER_H__
#define __OPENGL_RENDERER_H__

#include <openglframework.h>

using namespace openglframework;

class OpenGLRenderer
{
public:
    virtual void render(GLFWViewer* viewer, int width, int height) = 0;

};

#endif