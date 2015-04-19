#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <glm/glm.hpp>
#include "Simulation.h"

#include "FluidSimpleGravityForce.h"
#include "FluidBoundingBox.h"
#include "FluidBrick.h"
#include "Simulation.h"
#include "MathDefs.h"
#include <openglframework.h>

Simulation* g_simulation;
GLFWViewer* g_viewer;

void simulate();
void display(int width, int height);
void reshape(GLFWwindow* window, int width, int height);
void mouseButton(GLFWwindow* window, int button, int action, int mods);
void mouseMotion(GLFWwindow* window, double x, double y);
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
void errorCallback(int error, const char* description);
void init();


void testBasicSetup(){
    // I guess.... try initializing a scene? 


    FluidSimpleGravityForce* sgf = new FluidSimpleGravityForce(-10.1, .0, .0);
    //FluidSimpleGravityForce* sgff = new FluidSimpleGravityForce(Vector3s(.3, .2, .1));

    Scene scene;
    scene.insertFluidForce(sgf);
    //scene.insertFluidForce(sgff);

    FluidBoundingBox fbox(-0, 10, -0, 10, -0, 10); 

    Fluid *fluid = new Fluid(1000, 2.0, 10000.0, .5, 3, 20, 3);

    //fluid.setFPMass(2.0);
    //fluid.setRestDensity(1.0);
    float x; 
    float y; 
    float z;
    for(int i = 0; i < 1000; ++i){
        x = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
        y = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
        z = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
        fluid->setFPPos(i, Vector3s(x, y, z));
        fluid->setFPVel(i, Vector3s(0, 0, 0));
    }
   //fluid->setFPPos(1, Vector3s(.2, .2, .1));
    //fluid->setFPVel(1, Vector3s(-.1, 0, 0));
    fluid->setBoundingBox(fbox);
    
    // printVec3(Vector3s(-0.3, 1, 3));

    std::cout << "adding fluid to scene" << std::endl;
  
    scene.insertFluid(fluid);    

    FluidBrick *fbrick = new FluidBrick(0, 1, 0, 1, 0, 1); 
    scene.insertFluidBoundary(fbrick); 
    
    Stepper stepper;   
    
    stepper.stepScene(scene, .01);

    //FluidBoundingBox fbox; 
//    std::cout << fbox.minX() << std::endl;

    std::cout << "end test" << std::endl;    



}


int main(void)
{

    // Wow this is going to be my terrible, terrible 'test' function thing

    testBasicSetup();

    g_viewer = new GLFWViewer();
    Vector2 windowsSize = Vector2(600, 400);
    bool initOK = g_viewer->init("OpenGL Framework Demo", windowsSize);
    if (!initOK) return 1;

    Scene *scene = new Scene();

    FluidSimpleGravityForce* sgf = new FluidSimpleGravityForce(-10.1, .0, .0);
    scene->insertFluidForce(sgf);

    FluidBoundingBox *fbox = new FluidBoundingBox(-0, 10, -0, 10, -0, 10);

    Fluid *fluid = new Fluid(1, 2.0, 1.0, 1.4, 3, 10);

    //fluid.setFPMass(2.0);
    //fluid.setRestDensity(1.0);
    fluid->setFPPos(0, Vector3s(1.01, 0, 0));
    fluid->setFPVel(0, Vector3s(-10, 0, 0));
    //fluid->setFPPos(1, Vector3s(.2, .2, .1));
    //fluid->setFPVel(1, Vector3s(-.1, 0, 0));
    fluid->setBoundingBox(*fbox);

    scene->insertFluid(fluid);    

    FluidBrick *fbrick = new FluidBrick(0, 1, 0, 1, 0, 1); 
    scene->insertFluidBoundary(fbrick); 

    Stepper *stepper = new Stepper();
    
    // stepper.stepScene(scene, .01);

    SceneRenderer *renderer = new SceneRenderer(scene);

    g_simulation = new Simulation(scene, stepper, renderer);



    g_viewer->bindReshapeCallback(reshape);
    g_viewer->bindKeyboardCallback(keyboard);
    g_viewer->bindMouseButtonCallback(mouseButton);
    g_viewer->bindMouseMotionCallback(mouseMotion);
    g_viewer->bindDisplayCallback(display);
    g_viewer->bindIdleCallback(simulate);

    g_viewer->mainLoop();

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
    g_viewer->reshape(width, height);
}

// Called when a mouse button event occurs
void mouseButton(GLFWwindow* window, int button, int action, int mods) {
    g_viewer->mouseButtonEvent(button, action, mods);
}

// Called when a mouse motion event occurs
void mouseMotion(GLFWwindow* window, double x, double y) {
    g_viewer->mouseMotionEvent(x, y);
}

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
    g_viewer->keyboard(key, scancode, action, mods);
}

// Display the scene
void display(int width, int height) {

    g_simulation->display(g_viewer, width, height);
}

void errorCallback(int error, const char* description)
{
    fputs(description, stderr);
}

