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
    r = r * r * r; // (h^2 - r^2)^3
    return r * (315.0 / (64.0 * PI * h * h * h * h * h * h * h * h * h));
}


scalar wSpikyKernel(Vector3s &pi, Vector3s &pj, scalar h){
    scalar r = glm::distance(pi, pj); 
    if(r > h || r < 0)
        return 0;
    
    r = (h - r); 
    r = r * r * r;  // (h-r)^3
    return r * (15.0 / (PI * h * h * h * h * h * h)); 
}

scalar wViscosityKernel(Vector3s &pi, Vector3s &pj, scalar h){
    scalar r = glm::distance(pi, pj); 
    if(r > h || r < 0)
        return 0;
    
    scalar num = -(r*r*r)/(2.0*h*h*h) + (r*r)/(h*h) + h/(2.0*r) - 1; 
    return num * (15.0 / (2.0 * PI * h * h * h)); 
        
}


Vector3s wPoly6KernelGrad(Vector3s &pi, Vector3s &pj, scalar h){
    Vector3s dp = pi - pj; 
    scalar r = glm::length(dp);  
    if(r > h || r < 0)
        return Vector3s(0.0, 0.0, 0.0); 

    scalar scale = (h * h - r * r); 
    scale = scale * scale;  
    scale = (-945.0 / (32.0 * PI * h * h * h * h * h * h * h * h * h)); 
    return scale * dp; 
}

Vector3s wSpikyKernelGrad(Vector3s &pi, Vector3s &pj, scalar h){
    Vector3s dp = pi - pj; 
    scalar r = glm::length(dp);  
    if(r > h || r < 0)
        return Vector3s(0.0, 0.0, 0.0); 
    scalar scale = -45.0 / (PI * h * h * h * h * h * h) * (h - r) * (h - r); 
    return scale * dp; 
}


