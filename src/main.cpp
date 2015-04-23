#include "main.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "Simulation.h"
#include "FluidSimpleGravityForce.h"
#include "FluidBoundingBox.h"
#include "FluidBrick.h"
#include "MathDefs.h"
#include "StringUtilities.h"
#include <tclap/CmdLine.h>

void testBasicSetup(){
    // I guess.... try initializing a scene? 


    FluidSimpleGravityForce* sgf = new FluidSimpleGravityForce(-10.1, .0, .0);
    //FluidSimpleGravityForce* sgff = new FluidSimpleGravityForce(Vector3s(.3, .2, .1));

    Scene scene;
    scene.insertFluidForce(sgf);
    //scene.insertFluidForce(sgff);

    FluidBoundingBox fbox(-0, 10, -0, 10, -0, 10); 

    Fluid *fluid = new Fluid(1000, 2.0, 10000.0, .5, 3, 100, 3);

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

void parseCommandLine(int argc, char **argv) {

    try 
    {
      TCLAP::CmdLine cmd("Position-Based Fluid Sim");

      // XML scene file to load
      TCLAP::ValueArg<std::string> scene("s", "scene", "Simulation to run; an xml scene file", false, "", "string", cmd);

      // Begin the scene paused or running
      TCLAP::ValueArg<bool> paused("p", "paused", "Begin the simulation paused if 1, running if 0", false, true, "boolean", cmd);

      // Run the simulation with rendering enabled or disabled
      TCLAP::ValueArg<bool> display("d", "display", "Run the simulation with display enabled if 1, without if 0", false, true, "boolean", cmd);

      cmd.parse(argc, argv);

      g_xml_scene_file = scene.getValue();
      g_paused = paused.getValue();
      g_rendering_enabled = display.getValue();

    }
    catch (TCLAP::ArgException& e) 
    {
      std::cerr << "error: " << e.what() << std::endl;
      exit(1);
    }

}

void loadScene( const std::string& file_name) {

    // Maximum time in the simulation to run for. This has nothing to do with run time, cpu time, etc. This is time in the 'virtual world'.
    scalar max_time;
    // Maximum frequency, in wall clock time, to execute the simulation for. This serves as a cap for simulations that run too fast to see a solution.
    scalar steps_per_sec_cap = 100.0;

    // Load the simulation and pieces of rendring and UI state
    assert( g_simulation == NULL );
    // XML parse scene here
    // TwoDSceneXMLParser xml_scene_parser;
    // xml_scene_parser.loadExecutableSimulation( file_name, g_simulate_comparison, g_rendering_enabled, g_display_controller, &g_executable_simulation,
    //                                            view, g_dt, max_time, steps_per_sec_cap, g_bgcolor, g_description, g_scene_tag );

    //PLACEHOLDER
        Scene *scene = new Scene();

        FluidSimpleGravityForce* sgf = new FluidSimpleGravityForce(-10.1, .0, .0);
        scene->insertFluidForce(sgf);

        FluidBoundingBox fbox(-5, 10, -5, 10, -5, 10);

        Fluid *fluid = new Fluid(2000, 2.0, 100000.0, .5, 3, 100, 3);

         //fluid.setFPMass(2.0);
         //fluid.setRestDensity(1.0);
         float x; 
         float y; 
         float z;
         for(int i = 0; i < 2000; ++i){
             x = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
             y = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
             z = static_cast <float> (rand()) / static_cast<float>(RAND_MAX/9.0);
             fluid->setFPPos(i, Vector3s(x, y, z));
             fluid->setFPVel(i, Vector3s(0, 0, 0));
         }
        //fluid->setFPPos(1, Vector3s(.2, .2, .1));
         //fluid->setFPVel(1, Vector3s(-.1, 0, 0));
        fluid->setBoundingBox(fbox);

        scene->insertFluid(fluid);    

        FluidBrick *fbrick = new FluidBrick(0, 1, 0, 1, 0, 1); 
        scene->insertFluidBoundary(fbrick); 

        Stepper *stepper = new Stepper();
        
        // stepper.stepScene(scene, .01);

        SceneRenderer *renderer = NULL;
        if (g_rendering_enabled)
            renderer = new SceneRenderer(scene);

        g_simulation = new Simulation(scene, stepper, renderer);
        g_dt = 0.01;
        max_time = 10.0;
    //END PLACEHOLDER

    assert( g_simulation != NULL );

    // To cap the framerate, compute the minimum time a single timestep should take
    g_sec_per_frame = 1.0/steps_per_sec_cap;
    // Integer number of timesteps to take
    g_num_steps = ceil(max_time/g_dt);
    // We begin at the 0th timestep
    g_current_step = 0;
}

int main(int args, char **argv)
{
    parseCommandLine(args, argv);

    // Wow this is going to be my terrible, terrible 'test' function thing
    //testBasicSetup();

    if (g_rendering_enabled)
        initializeOpenGLandGLFW();


    loadScene(g_xml_scene_file);

    if (g_rendering_enabled)
        g_viewer->mainLoop();
    else
        headlessSimLoop();

    return 0;
}

void stepSystem() {

      // Determine if the simulation is complete
      if( g_current_step >= g_num_steps )
      {
        std::cout << outputmod::startpink << "PBF message: " << outputmod::endpink << "Simulation complete at time " << g_current_step*g_dt << ". Exiting." << std::endl;
        g_simulation_ran_to_completion = true;
        exit(0);
      }

      // Step the system forward in time
      g_simulation->stepSystem(g_dt);
      std::cout << outputmod::startgreen << "Time step: " << outputmod::endgreen << (g_current_step*g_dt) << std::endl;
      g_current_step++;
}

void headlessSimLoop() {

    scalar nextpercent = 0.02;
    std::cout << outputmod::startpink << "Progress: " << outputmod::endpink;
    for( int i = 0; i < 50; ++i ) std::cout << "-";
    std::cout << std::endl;
    std::cout << "          ";
    while( true )
    {
      scalar percent_done = ((double)g_current_step)/((double)g_num_steps);
      if( percent_done >= nextpercent )
      {
        nextpercent += 0.02;
        std::cout << "." << std::flush;
      }
      stepSystem();
    }
}

// idle function from GLFW
void idle() {

    //std::cout << "g_last_time: " << g_last_time << std::endl;
    // Trigger the next timestep
    double current_time = timingutils::seconds();
    //std::cout << "current_time: " << current_time << std::endl;
    //std::cout << "g_sec_per_frame: " << g_sec_per_frame << std::endl;
    if( !g_paused && current_time-g_last_time >= g_sec_per_frame ) 
    {
      g_last_time = current_time;
      stepSystem();
    }
}

// Initialization
void initializeOpenGLandGLFW() {

    g_viewer = new openglframework::GLFWViewer();
    openglframework::Vector2 windowsSize = openglframework::Vector2(600, 400);
    bool initOK = g_viewer->init("OpenGL Framework Demo", windowsSize);
    if (!initOK) {
        std::cerr << "Error initializing GLFW viewer" << std::endl;
        exit(1);
        return;
    }

    g_viewer->bindReshapeCallback(reshape);
    g_viewer->bindKeyboardCallback(keyboard);
    g_viewer->bindMouseButtonCallback(mouseButton);
    g_viewer->bindMouseMotionCallback(mouseMotion);
    g_viewer->bindDisplayCallback(display);
    g_viewer->bindIdleCallback(idle);

    // Define the background color (black)
    glClearColor(g_bgcolor.r, g_bgcolor.g, g_bgcolor.b, 1.0);
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

    if (key == GLFW_KEY_S && action == GLFW_PRESS) {
        stepSystem();
    }
    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        g_paused = !g_paused;
    }
}

// Display the scene
void display(int width, int height) {

    g_simulation->display(g_viewer, width, height);
}

void errorCallback(int error, const char* description)
{
    fputs(description, stderr);
}

