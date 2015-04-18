#ifndef GLFW_VIEWER_H
#define GLFW_VIEWER_H

// Libraries
#include "Shader.h"
#include "Camera.h"
#include "maths/Vector2.h"
#include <string>
#include <GLFW/glfw3.h>

namespace openglframework {

// Class Renderer
class GLFWViewer {

    private:

        GLFWwindow *m_window;

        void (*m_reshapeCallback) (GLFWwindow*, int, int);
        void (*m_mouseButtonCallback) (GLFWwindow*, int, int, int);
        void (*m_mouseMotionCallback) (GLFWwindow*, double, double);
        void (*m_keyboardCallback) (GLFWwindow*, int, int, int, int);
        void (*m_idleCallback) ();
        void (*m_displayCallback) (int, int);

        // -------------------- Attributes -------------------- //

        // Camera
        Camera mCamera;

        // Center of the scene
        Vector3 mCenterScene;

        // Last mouse coordinates on the windows
        int mLastMouseX, mLastMouseY;

        // Last point computed on a sphere (for camera rotation)
        Vector3 mLastPointOnSphere;

        // True if the last point computed on a sphere (for camera rotation) is valid
        bool mIsLastPointOnSphereValid;

        // -------------------- Methods -------------------- //

        // Initialize the GLFW library
        bool initGLFW(const std::string& windowsTitle,
                      const Vector2& windowsSize);

        bool mapMouseCoordinatesToSphere(int xMouse, int yMouse, Vector3& spherePoint) const;


    public:

        // -------------------- Methods -------------------- //

        // Constructor
        GLFWViewer();

        // Destructor
        ~GLFWViewer();

        // Initialize the viewer
        bool init(const std::string& windowsTitle,
                  const Vector2& windowsSize);

        void mainLoop();

        // GLFW callbacks
        void bindReshapeCallback(void (*reshapeCallback) (GLFWwindow*, int, int));
        void bindMouseButtonCallback(void (*mouseButtonCallback) (GLFWwindow*, int, int, int));
        void bindMouseMotionCallback(void (*mouseMotionCallback) (GLFWwindow*, double, double));
        void bindKeyboardCallback(void (*keyboardCallback) (GLFWwindow*, int, int, int, int));
        void bindIdleCallback(void (*idleCallback) ());
        void bindDisplayCallback(void (*displayCallback) (int, int));

        // Called when the windows is reshaped
        void reshape(int width, int height);

        // Called when a GLFW mouse button event occurs
        void mouseButtonEvent(int button, int action, int mods);
        // Called when a GLFW mouse motion event occurs
        void mouseMotionEvent(double xMouse, double yMouse);

        void keyboard(int key, int scancode, int action, int mods);

        void idle();
        void display();

        // Set the scene position (where the camera needs to look at)
        void setScenePosition(const Vector3& position, float sceneRadius);

        // Set the camera so that we can view the whole scene
        void resetCameraToViewAll();

        // Zoom the camera
        void zoom(int xMouse, int yMouse);

        // Translate the camera
        void translate(int xMouse, int yMouse);

        // Rotate the camera
        void rotate(int xMouse, int yMouse);

        // Get the camera
        Camera& getCamera();

};

// Set the scene position (where the camera needs to look at)
inline void GLFWViewer::setScenePosition(const Vector3& position, float sceneRadius) {

    // Set the position and radius of the scene
    mCenterScene = position;
    mCamera.setSceneRadius(sceneRadius);

    // Reset the camera position and zoom in order to view all the scene
    resetCameraToViewAll();
}

// Get the camera
inline Camera& GLFWViewer::getCamera() {
   return mCamera;
}

}
#endif
