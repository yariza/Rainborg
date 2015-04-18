#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <glm/glm.hpp>
#include "Simulation.h"

#include "FluidSimpleGravityForce.h"
#include "FluidBoundingBox.h"
#include "MathDefs.h"

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

int foo()
{
        glm::vec4 Position = glm::vec4(glm::vec3(0.0), 1.0);
        glm::mat4 Model = glm::mat4(1.0);
        Model[3] = glm::vec4(1.0, 1.0, 0.0, 1.0);
        glm::vec4 Transformed = Model * Position;
        return 0;
}

void testBasicSetup(){
    // I guess.... try initializing a scene? 


    FluidSimpleGravityForce* sgf = new FluidSimpleGravityForce(-1.1, .2, .3);
    FluidSimpleGravityForce* sgff = new FluidSimpleGravityForce(Vector3s(.3, .2, .1));

    Scene scene;
    scene.insertFluidForce(sgf);
    scene.insertFluidForce(sgff);

    FluidBoundingBox fbox(-1, 4.3, -1.2, 5.4, -1.1, 5.5); 

    Fluid *fluid = new Fluid(2, 2.0, 1.0, 1.4, 3, 10);

    //fluid.setFPMass(2.0);
    //fluid.setRestDensity(1.0);
    fluid->setFPPos(0, Vector3s(1, 2.1, 3));
    fluid->setFPVel(0, Vector3s(1.1, .4, .2));
    fluid->setFPPos(1, Vector3s(3.2, -.2, 1));
    fluid->setFPVel(1, Vector3s(-.3, .2, .1));
    fluid->setBoundingBox(fbox);
    
    // printVec3(Vector3s(-0.3, 1, 3));

    std::cout << "adding fluid to scene" << std::endl;
  
    scene.insertFluid(fluid);    
    
    Stepper stepper;   
    
    stepper.stepScene(scene, .01);

//    FluidBoundingBox fbox; 
//    std::cout << fbox.minX() << std::endl;

    std::cout << "end test" << std::endl;    



}


int main(void)
{

    // Wow this is going to be my terrible, terrible 'test' function thing

    testBasicSetup();
    

/*
    std::cout << foo() << std::endl;

    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current 
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    // Loop until the user closes the window 
    while (!glfwWindowShouldClose(window))
    {
        // Render here 
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
        glBegin(GL_TRIANGLES);
            glColor3f(1.f, 0.f, 0.f);
            glVertex3f(-0.6f, -0.4f, 0.f);
            glColor3f(0.f, 1.f, 0.f);
            glVertex3f(0.6f, -0.4f, 0.f);
            glColor3f(0.f, 0.f, 1.f);
            glVertex3f(0.f, 0.6f, 0.f);
        glEnd();

        // Swap front and back buffers 
        glfwSwapBuffers(window);

        //  Poll for and process events 
        glfwPollEvents();
    }


    glfwDestroyWindow(window);
    glfwTerminate();

*/
    return 0;
}

