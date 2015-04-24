#include <stdlib.h>
#include <openglframework.h>
#include "Simulation.h"
#include "TimingUtilities.h"

// callback functions for GLFW
void idle();
void display(int width, int height);
void reshape(GLFWwindow* window, int width, int height);
void mouseButton(GLFWwindow* window, int button, int action, int mods);
void mouseMotion(GLFWwindow* window, double x, double y);
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
void errorCallback(int error, const char* description);

// other functions
void initializeOpenGLandGLFW();
void headlessSimLoop();
void stepSystem();

// Global variables
Simulation* g_simulation;
openglframework::GLFWViewer* g_viewer;
openglframework::Color g_bgcolor(0.0, 0.0, 0.0, 1.0);

bool g_rendering_enabled = true;
double g_sec_per_frame;
double g_last_time = timingutils::seconds();
int g_num_steps = 0;
scalar g_dt = 0.0;

// Simulation state
bool g_paused = true;
int g_current_step = 0;
bool g_simulation_ran_to_completion = false;

// Parser state
std::string g_xml_scene_file;

// gpu mode?
bool g_gpu_mode;
