#ifndef __FLUID_VOLUME_H__
#define __FLUID_VOLUME_H__

#include "MathDefs.h"

typedef enum {
    kFLUID_VOLUME_MODE_BOX,
    kFLUID_VOLUME_MODE_SPHERE
} fluid_volume_mode_t;

struct FluidVolume {

    scalar m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ;
    scalar m_dens_cbrt; // cache cbrt of density for later
    int m_numParticles;
    fluid_volume_mode_t m_mode;
    bool m_random;
    Vector4s m_color;

    FluidVolume();
    FluidVolume(scalar minX, scalar maxX, scalar minY, scalar maxY, scalar minZ, scalar maxZ,
                int numParticles, fluid_volume_mode_t mode, bool random, scalar r=(81.0/255.0), scalar g=(163.0/255.0), scalar b=(255.0/255.0), scalar alpha = .5f);
    FluidVolume(const FluidVolume& otherVolume);
    
    void setParticlePositions(Vector3s* pos, int offset);
    int setSpacing(scalar h);
    void setParticleColors(Vector4s *pos, int offset);
};

#endif
