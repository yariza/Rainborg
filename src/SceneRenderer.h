#ifndef __SCENE_RENDERER_H__
#define __SCENE_RENDERER_H__

#include <openglframework.h>
#include "Scene.h"
#include "FluidRenderer.h"

class SceneRenderer : OpenGLRenderer
{
public:
    SceneRenderer(Scene* scene);
    ~SceneRenderer();

    virtual void render(openglframework::GLFWViewer* viewer, int width, int height);
    void toggleDebugMode();

private:
    Scene* m_scene;
    std::vector<FluidRenderer*> m_renderers;
    bool m_debug;
};

#endif
