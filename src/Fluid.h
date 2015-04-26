#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>
#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"

class Scene;

class Fluid {

public:
    virtual ~Fluid();

    virtual void stepSystem(Scene& scene, scalar dt) = 0;

    virtual void loadFluidVolumes() = 0;

    virtual void setFPMass(scalar fpm) = 0;
    virtual void setRestDensity(scalar p0) = 0;
    virtual void setKernelH(scalar h) = 0;
    virtual void setNumIterations(int iter) = 0;
    virtual void setBoundingBox(FluidBoundingBox& newBound) = 0;

    virtual void insertFluidVolume(FluidVolume& volume) = 0;

    virtual int getNumParticles() const = 0;
    virtual int getNumIterations() const = 0;
    virtual int getMaxNeighbors() const = 0;
    virtual int getMinNeighbors() const = 0;
    virtual scalar getFPMass() const = 0;
    virtual scalar getRestDensity() const = 0;
    virtual scalar getKernelH() const = 0;
    virtual const FluidBoundingBox& getBoundingBox() const = 0;
    virtual const std::vector<FluidVolume>& getFluidVolumes() const = 0;

protected:

};

#endif