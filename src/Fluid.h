#ifndef __FLUID_H__
#define __FLUID_H__

#include <vector>
#include "MathDefs.h"
#include "FluidBoundingBox.h"
#include "FluidVolume.h"

class Scene;

class Fluid {

public:
    Fluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors);
    Fluid(const Fluid& otherFluid);
    virtual ~Fluid();

    virtual void stepSystem(Scene& scene, scalar dt) = 0;

    virtual void loadFluidVolumes() = 0;

    virtual void setFPMass(scalar fpm);
    virtual void setRestDensity(scalar p0);
    virtual void setKernelH(scalar h);
    virtual void setNumIterations(int iter);
    virtual void setBoundingBox(FluidBoundingBox& newBound);

    virtual void insertFluidVolume(FluidVolume& volume);

    virtual int getNumParticles() const;
    virtual int getNumIterations() const;
    virtual int getMaxNeighbors() const;
    virtual int getMinNeighbors() const;
    virtual scalar getFPMass() const;
    virtual scalar getRestDensity() const;
    virtual scalar getKernelH() const;
    virtual const FluidBoundingBox& getBoundingBox() const;
    virtual const std::vector<FluidVolume>& getFluidVolumes() const;

    virtual void updateVBO(float* dptr) = 0;

protected:

    scalar m_fpmass;
    scalar m_p0;
    scalar m_h;
    int m_iters;
    int m_maxNeighbors;
    int m_minNeighbors;

    std::vector<FluidVolume> m_volumes;
    FluidBoundingBox m_boundingBox; 

};

#endif
