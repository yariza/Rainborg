#include "MathDefs.h"
#include <iostream>

void printVec3(Vector3s vec) {
    std::cout << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
}

scalar wPoly6Kernel(Vector3s &pi, Vector3s &pj, scalar h){
    scalar r = glm::distance(pi, pj); 
    if(r > h || r < 0)
        return 0; 

    r = ((h * h) - (r * r)); 
    r = r * r * r; 
    return r * (315.0 / (64.0 * PI * h * h * h * h));
}


scalar wSpikyKernel(Vector3s &pi, Vector3s &pj, scalar h){
return 0; 
}

Vector3s wPoly6KernelGrad(Vector3s &pi, Vector3s &pj, scalar h){


}

Vector3s wSpikyKernelGrad(Vector3s &pi, Vector3s &pj, scalar h){

}


