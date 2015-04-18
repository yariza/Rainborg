#ifndef __MATH_DEFS_H__
#define __MATH_DEFS_H__

#include <glm/glm.hpp>

typedef float scalar;  // so that we can deal with scalar without worrying about the underlying representation
typedef glm::vec3 Vector3s; 
typedef glm::vec4 Vector4s; 

void printVec3(Vector3s vec);

//float wPoly6Kernel(Vector3s pi, Vector3s pj, scalar h); 




#endif
