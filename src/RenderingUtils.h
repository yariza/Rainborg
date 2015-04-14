#ifndef __RENDERING_UTILS_H__
#define __RENDERING_UTILS_H__ value

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include "StringUtilities.h"
#include <iostream>
#include <cstdio>

namespace renderingutils
{

// False => error
bool checkGLErrors();


}

#endif