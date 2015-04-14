#include "RenderingUtils.h"

namespace renderingutils
{

bool checkGLErrors()
{
  GLenum errCode;
  const GLubyte *errString;

  if ((errCode = glGetError()) != GL_NO_ERROR) 
  {
    errString = gluErrorString(errCode);
    std::cout << outputmod::startred << "OpenGL Error:" << outputmod::endred << std::flush;
    fprintf(stderr, " %s\n", errString);
    return false;
  }
  return true;
}

}
