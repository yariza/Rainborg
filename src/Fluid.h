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
    virtual void setBoundingBox(FluidBoundingBox* newBound);

    virtual void insertFluidVolume(FluidVolume& volume);

    virtual int getNumParticles() const;
    virtual int getNumIterations() const;
    virtual int getMaxNeighbors() const;
    virtual int getMinNeighbors() const;
    virtual scalar getFPMass() const;
    virtual scalar getRestDensity() const;
    virtual scalar getKernelH() const;
    virtual const FluidBoundingBox* getBoundingBox() const;
    virtual const std::vector<FluidVolume>& getFluidVolumes() const;

    virtual void updateVBO(float* dptr) = 0;
    virtual Vector4s* getColors() const;


protected:
 
    scalar m_fpmass; // 'mass' per particle
    scalar m_p0; // rest density
    scalar m_h; // kernel width
    int m_iters; // number of iterations through constraint solver
    int m_maxNeighbors; // max number of expected neighbors
    int m_minNeighbors; // min number of neighbors to use calculations

    std::vector<FluidVolume> m_volumes; // initial volumes from fluid parsing
    FluidBoundingBox* m_boundingBox; // the boundary associated with this fluid
    Vector4s *m_colors; // colors!


};

#endif
