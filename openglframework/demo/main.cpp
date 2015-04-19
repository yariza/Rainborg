/********************************************************************************
* OpenGL-Framework                                                              *
* Copyright (c) 2013 Daniel Chappuis                                            *
*********************************************************************************
*                                                                               *
* This software is provided 'as-is', without any express or implied warranty.   *
* In no event will the authors be held liable for any damages arising from the  *
* use of this software.                                                         *
*                                                                               *
* Permission is granted to anyone to use this software for any purpose,         *
* including commercial applications, and to alter it and redistribute it        *
* freely, subject to the following restrictions:                                *
*                                                                               *
* 1. The origin of this software must not be misrepresented; you must not claim *
*    that you wrote the original software. If you use this software in a        *
*    product, an acknowledgment in the product documentation would be           *
*    appreciated but is not required.                                           *
*                                                                               *
* 2. Altered source versions must be plainly marked as such, and must not be    *
*    misrepresented as being the original software.                             *
*                                                                               *
* 3. This notice may not be removed or altered from any source distribution.    *
*                                                                               *
********************************************************************************/

// Libraries
#include "Scene.h"
#include "stdlib.h"
#include "stdio.h"

// Declarations
void simulate();
void display(int width, int height);
void reshape(GLFWwindow* window, int width, int height);
void mouseButton(GLFWwindow* window, int button, int action, int mods);
void mouseMotion(GLFWwindow* window, double x, double y);
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
void errorCallback(int error, const char* description);
void init();

// Namespaces
using namespace openglframework;

// Global variables
GLFWViewer* viewer;
Scene* scene;

// Main function
int main(int argc, char** argv) {

    // Create and initialize the Viewer
    viewer = new GLFWViewer();
    Vector2 windowsSize = Vector2(600, 400);
    bool initOK = viewer->init("OpenGL Framework Demo", windowsSize);
    if (!initOK) return 1;

    // Create the scene
    scene = new Scene(viewer);

    init();

    viewer->bindReshapeCallback(reshape);
    viewer->bindDisplayCallback(display);
    viewer->bindIdleCallback(simulate);
    viewer->bindKeyboardCallback(keyboard);
    viewer->bindMouseButtonCallback(mouseButton);
    viewer->bindMouseMotionCallback(mouseMotion);

    viewer->mainLoop();

    delete viewer;
    delete scene;

    return 0;
}

// Simulate function
void simulate() {

    // Display the scene
    // display();
}

// Initialization
void init() {

    // Define the background color (black)
    glClearColor(0.0, 0.0, 0.0, 1.0);
}

// Reshape function
void reshape(GLFWwindow* window, int width, int height) {
    viewer->reshape(width, height);
}

// Called when a mouse button event occurs
void mouseButton(GLFWwindow* window, int button, int action, int mods) {
    viewer->mouseButtonEvent(button, action, mods);
}

// Called when a mouse motion event occurs
void mouseMotion(GLFWwindow* window, double x, double y) {
    viewer->mouseMotionEvent(x, y);
}

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
    viewer->keyboard(key, scancode, action, mods);
}

// Display the scene
void display(int width, int height) {

    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scene->display();
}

void errorCallback(int error, const char* description)
{
    fputs(description, stderr);
}

