#include "GLFWViewer.h"
#include <iostream>

using namespace openglframework;
using namespace std;

GLFWViewer::GLFWViewer()
: m_window(NULL)
, m_reshapeCallback(NULL)
, m_keyboardCallback(NULL)
, m_mouseButtonCallback(NULL)
, m_mouseMotionCallback(NULL) {

}

GLFWViewer::~GLFWViewer() {

}

bool GLFWViewer::init(const string& windowsTitle,
                      const Vector2& windowsSize) {

    bool outputValue = initGLFW(windowsTitle, windowsSize);

    // Active the multi-sampling by default
    // if (isMultisamplingActive) {
    //     activateMultiSampling(true);
    // }

    // init GLEW
    GLenum error = glewInit();
    if (error != GLEW_OK) {

        // Problem: glewInit failed, something is wrong
        cerr << "GLEW Error : " << glewGetErrorString(error) << std::endl;
        assert(false);
        return false;
    }

    return outputValue;
}

bool GLFWViewer::initGLFW(const string& windowsTitle,
                          const Vector2& windowsSize) {

    if (!glfwInit()) return false;

    m_window = glfwCreateWindow(windowsSize.x, windowsSize.y, windowsTitle.c_str(), NULL, NULL);

    if (!m_window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    float radius = 10.0f;
    Vector3 center(0.0, 0.0, 0.0);

    setScenePosition(center, radius);

    return true;
}

void GLFWViewer::bindReshapeCallback(void (*reshapeCallback) (GLFWwindow*, int, int) ) {
    m_reshapeCallback = reshapeCallback;
}

void GLFWViewer::bindMouseButtonCallback(void (*mouseButtonCallback) (GLFWwindow*, int, int, int) ) {
    m_mouseButtonCallback = mouseButtonCallback;
}

void GLFWViewer::bindMouseMotionCallback(void (*mouseMotionCallback) (GLFWwindow*, double, double) ) {
    m_mouseMotionCallback = mouseMotionCallback;
}

void GLFWViewer::bindKeyboardCallback(void (*keyboardCallback) (GLFWwindow*, int, int, int, int) ) {
    m_keyboardCallback = keyboardCallback;
}

void GLFWViewer::bindIdleCallback(void (*idleCallback) ()) {
    m_idleCallback = idleCallback;
}

void GLFWViewer::bindDisplayCallback(void (*displayCallback) (int, int)) {
    m_displayCallback = displayCallback;
}

void GLFWViewer::reshape(int width, int height) {
    mCamera.setDimensions(width, height);
    glViewport(0, 0, width, height);

    display();
}

void GLFWViewer::keyboard(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GL_TRUE);
    }
}

void GLFWViewer::mouseButtonEvent(int button, int action, int mods) {

    double x, y;
    glfwGetCursorPos(m_window, &x, &y);

    // If the mouse button is pressed
    if (action == GLFW_PRESS) {
        mLastMouseX = x;
        mLastMouseY = y;
        mIsLastPointOnSphereValid = mapMouseCoordinatesToSphere(x, y, mLastPointOnSphere);
    }
    else {  // If the mouse button is released
        mIsLastPointOnSphereValid = false;

        // If it is a mouse wheel click event
        if (button == 3) {
            zoom(0, (int) (y - 0.05f * mCamera.getWidth()));
        }
        else if (button == 4) {
            zoom(0, (int) (y + 0.05f * mCamera.getHeight()));
        }
    }
}

void GLFWViewer::mouseMotionEvent(double xMouse, double yMouse) {

    // Zoom
    if ((glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS &&
         glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) ||
        (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS &&
         glfwGetKey(m_window, GLFW_KEY_LEFT_ALT))) {
        zoom(xMouse, yMouse);
    }
    // Translation
    else if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS ||
             glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
             (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) &&
             glfwGetKey(m_window, GLFW_KEY_LEFT_ALT))) {
        translate(xMouse, yMouse);
    }
    // Rotation
    else if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT)) {
        rotate(xMouse, yMouse);
    }

    // Remember the mouse position
    mLastMouseX = xMouse;
    mLastMouseY = yMouse;
    mIsLastPointOnSphereValid = mapMouseCoordinatesToSphere(xMouse, yMouse, mLastPointOnSphere);
}

void GLFWViewer::idle() {
    if (m_idleCallback)
        m_idleCallback();
}

void GLFWViewer::display() {
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    if (m_displayCallback)
        m_displayCallback(width, height);
}

void GLFWViewer::mainLoop() {

    if (m_reshapeCallback)
        glfwSetFramebufferSizeCallback(m_window, m_reshapeCallback);
    if (m_keyboardCallback)
        glfwSetKeyCallback(m_window, m_keyboardCallback);
    if (m_mouseButtonCallback)
        glfwSetMouseButtonCallback(m_window, m_mouseButtonCallback);
    if (m_mouseMotionCallback)
        glfwSetCursorPosCallback(m_window, m_mouseMotionCallback);

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    reshape(width, height);

    while (!glfwWindowShouldClose(m_window)) {

        display();
        idle();
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

// Set the camera so that we can view the whole scene
void GLFWViewer::resetCameraToViewAll() {

    // Move the camera to the origin of the scene
    mCamera.translateWorld(-mCamera.getOrigin());

    // Move the camera to the center of the scene
    mCamera.translateWorld(mCenterScene);

    // Set the zoom of the camera so that the scene center is
    // in negative view direction of the camera
    mCamera.setZoom(1.0);
}

// Map the mouse x,y coordinates to a point on a sphere
bool GLFWViewer::mapMouseCoordinatesToSphere(int xMouse, int yMouse, Vector3& spherePoint) const {

    int width = mCamera.getWidth();
    int height = mCamera.getHeight();

    if ((xMouse >= 0) && (xMouse <= width) && (yMouse >= 0) && (yMouse <= height)) {
        float x = float(xMouse - 0.5f * width) / float(width);
        float y = float(0.5f * height - yMouse) / float(height);
        float sinx = sin(PIE * x * 0.5f);
        float siny = sin(PIE * y * 0.5f);
        float sinx2siny2 = sinx * sinx + siny * siny;

        // Compute the point on the sphere
        spherePoint.x = sinx;
        spherePoint.y = siny;
        spherePoint.z = (sinx2siny2 < 1.0) ? sqrt(1.0f - sinx2siny2) : 0.0f;

        return true;
    }

    return false;
}

// Zoom the camera
void GLFWViewer::zoom(int xMouse, int yMouse) {
    float dy = static_cast<float>(yMouse - mLastMouseY);
    float h = static_cast<float>(mCamera.getHeight());

    // Zoom the camera
    mCamera.setZoom(-dy / h);
}

// Translate the camera
void GLFWViewer::translate(int xMouse, int yMouse) {
   float dx = static_cast<float>(xMouse - mLastMouseX);
   float dy = static_cast<float>(yMouse - mLastMouseY);

   // Translate the camera
   mCamera.translateCamera(-dx / float(mCamera.getWidth()),
                           -dy / float(mCamera.getHeight()), mCenterScene);
}

// Rotate the camera
void GLFWViewer::rotate(int xMouse, int yMouse) {
    if (mIsLastPointOnSphereValid) {

        Vector3 newPoint3D;
        bool isNewPointOK = mapMouseCoordinatesToSphere(xMouse, yMouse, newPoint3D);

        if (isNewPointOK) {
            Vector3 axis = mLastPointOnSphere.cross(newPoint3D);
            float cosAngle = mLastPointOnSphere.dot(newPoint3D);

            float epsilon = std::numeric_limits<float>::epsilon();
            if (fabs(cosAngle) < 1.0f && axis.length() > epsilon) {
                axis.normalize();
                float angle = 2.0f * acos(cosAngle);

                // Rotate the camera around the center of the scene
                mCamera.rotateAroundLocalPoint(axis, -angle, mCenterScene);
            }
        }
    }
}

int GLFWViewer::getWindowWidth() {
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    return width;
}

int GLFWViewer::getWindowHeight() {
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    return height;
}
