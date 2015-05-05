#include "SerialFluid.h"
#include "Scene.h"

SerialFluid::SerialFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors)
: Fluid(mass, p0, h, iters, maxNeighbors, minNeighbors)
, m_eps(.01) 
{
    // DEFER ALL LOADING TO Fluid.loadFluidVolumes()
    m_grid = NULL; // don't know bounding box yet
    m_gridCount = NULL;
}


SerialFluid::SerialFluid(const SerialFluid& otherFluid)
: Fluid(otherFluid)
, m_eps(.01)
{
    // loadFluidVolumes is necessary to reallocate stuff again

	// Get grid size from bounding box, partitioned by kernel width
    m_gridX = ceil(m_boundingBox->width()/m_h);
    m_gridY = ceil(m_boundingBox->height()/m_h);
    m_gridZ = ceil(m_boundingBox->depth()/m_h);

    m_grid = (int *)malloc(m_gridX * m_gridY * m_gridZ * m_maxNeighbors * sizeof(int)); 
    m_gridCount = (int *)malloc(m_gridX * m_gridY * m_gridZ * sizeof(int)); 

    m_volumes = otherFluid.getFluidVolumes();

    assert(m_grid != NULL);
    assert(m_gridCount != NULL);
}

SerialFluid::~SerialFluid(){
    free(m_pos);
    free(m_ppos);
    free(m_lambda);
    free(m_pcalc); 
    free(m_dpos);
    free(m_vel);
    free(m_accumForce);
    free(m_colors); 
    free(m_gridInd);
    if(m_grid != NULL)
        free(m_grid);
    if(m_gridCount != NULL)
        free(m_gridCount);
}

void SerialFluid::setFPVel(int fp, const Vector3s& vel){
    // assert(fp >= 0 && fp < m_numParticles);
    m_vel[fp] = vel;  
}

void SerialFluid::setBoundingBox(FluidBoundingBox* bound){
    Fluid::setBoundingBox(bound);

    std::cout << "bounding box" << std::endl;

    if(m_grid != NULL)
        free(m_grid);
    if(m_gridCount != NULL)
        free(m_gridCount);

    m_gridX = ceil(m_boundingBox->width()/m_h);
    m_gridY = ceil(m_boundingBox->height()/m_h);
    m_gridZ = ceil(m_boundingBox->depth()/m_h);
    m_grid = (int *)malloc(m_gridX * m_gridY * m_gridZ * m_maxNeighbors * sizeof(int)); 
    m_gridCount = (int *)malloc(m_gridX * m_gridY * m_gridZ * sizeof(int)); 

    assert(m_grid != NULL);
    assert(m_gridCount != NULL);
}

void SerialFluid::setColor(int i, const Vector4s& col){
    m_colors[i] = col; 
}

Vector3s* SerialFluid::getFPPos() const{
    return m_pos; 
}

Vector3s* SerialFluid::getFPVel() const{
    return m_vel;
}

void SerialFluid::loadFluidVolumes() {

    int numParticles = getNumParticles();
    std::cout << "num particles: " << numParticles << std::endl;

    m_lambda = (scalar *)malloc(numParticles * sizeof(scalar)); 
    m_pcalc = (scalar *)malloc(numParticles * sizeof(scalar)); 
    memset(m_pcalc, 0, numParticles * sizeof(scalar));
    m_pos = (Vector3s *)malloc(numParticles * sizeof(Vector3s));
    m_ppos = (Vector3s *)malloc(numParticles * sizeof(Vector3s)); 
    m_dpos = (Vector3s *)malloc(numParticles * sizeof(Vector3s));
    m_vel = (Vector3s *)malloc(numParticles * sizeof(Vector3s)); 
    memset(m_vel, 0, numParticles * sizeof(Vector3s));
    m_accumForce = (Vector3s *)malloc(numParticles * sizeof(Vector3s)); 
    m_colors = (Vector4s *)malloc(numParticles * sizeof(Vector4s)); 
    m_gridInd = (int *)malloc(numParticles * 3 * sizeof(int));

    int offset = 0;
    for (std::vector<FluidVolume>::size_type i=0; i<m_volumes.size(); i++) {

        FluidVolume& volume = m_volumes[i];

        volume.setParticlePositions(m_pos, offset);
        volume.setParticleColors(m_colors, offset);

        offset += volume.m_numParticles;
    }
}

// Calculate positions, velocities for next time step
void SerialFluid::stepSystem(Scene& scene, scalar dt){

	// Accumulate forces in scene
    accumulateForce(scene); 

	// Apply forces to velocity
    updateVelocityFromForce(dt); 
    
	// Predict positions by applying velocity to current positions
    updatePredPosition(dt); 
    
    // Make sure that predicted positions don't go out of bounds here 
    memset(m_dpos, 0, getNumParticles() * sizeof(Vector3s)); 
    preserveOwnBoundary(); 
    
	// Update predicted positions to stay in bounds
    applydPToPredPos();
    buildGrid();   // Or at least, since neighbors are just adjacent grids, build grid structure

    // Loop for solve iterations
    for(int loop = 0; loop < m_iters; ++loop){
		// Fluids have constraint to maintain constant pressure
		// Calculate necessary movement to best handle constraint
        calculatePressures(); 
        calculateLambdas();  
        calculatedPos(); 

		// Resolve collisions with objects in scene and boundary
        dealWithCollisions(scene); 
        preserveOwnBoundary(); 

        // Update predicted position with dP
        applydPToPredPos(); 
    }

    // Update velocities
    recalculateVelocity(dt); 

    updateFinalPosition(); 
}

void SerialFluid::calculatePressures(){
    int gi = 0; // current grid id
    scalar press; 
    int ncount; // number of neighbors
    for(int p = 0; p < getNumParticles(); ++p){
        // Calculate pressure at each particle by grabbing all neighbors, determining their distance and mass
        ncount = 0; 
        press = 0;
        for(int i = std::max(0, m_gridInd[p*3]-1); i <= std::min(m_gridX-1, m_gridInd[p*3]+1); ++i){
            for(int j = std::max(0, m_gridInd[p*3+1]-1); j <= std::min(m_gridY-1, m_gridInd[p*3+1]+1); ++j){
                for(int k = std::max(0, m_gridInd[p*3+2]-1); k <= std::min(m_gridZ-1, m_gridInd[p*3+2]+1); ++k){
                    gi = getGridIdx(i, j, k); 
                    for(int n = 0; n < m_gridCount[gi]; ++n){ // for all particles in the grid
                        scalar pressN = wPoly6Kernel(m_ppos[p], m_ppos[m_grid[gi * m_maxNeighbors + n]], m_h); 
                        press += pressN;
                        if(pressN > 0)
                            ++ ncount; 
                    }            
                }
            }
        }        
        if(ncount <= m_minNeighbors && m_pcalc[p] == 0) // don't count self
            m_pcalc[p] = m_p0; 
        else 
            m_pcalc[p] = m_fpmass * press; // Wow I totally forgot that
    }
}

Vector3s SerialFluid::calcGradConstraint(Vector3s& pi, Vector3s& pj){
    return (scalar)(m_fpmass) * wSpikyKernelGrad(pi, pj, m_h) / (- m_p0); 
}

Vector3s SerialFluid::calcGradConstraintAtI(int p){
    Vector3s sumGrad(0.0, 0.0, 0.0); 
    // For neighbors, sum wSpikyKernelGrad(pi, pj, m_h)
    int gi; 
    for(int i = std::max(0, m_gridInd[p*3]-1); i <= std::min(m_gridX-1, m_gridInd[p*3]+1); ++i){
        for(int j = std::max(0, m_gridInd[p*3+1]-1); j <= std::min(m_gridY-1, m_gridInd[p*3+1]+1); ++j){
            for(int k = std::max(0, m_gridInd[p*3+2]-1); k <= std::min(m_gridZ-1, m_gridInd[p*3+2]+1); ++k){
                gi = getGridIdx(i, j, k); // current grid
                for(int n = 0; n < m_gridCount[gi]; ++n){ // for all particles in the grid
                    sumGrad += wSpikyKernelGrad(m_ppos[p], m_ppos[m_grid[gi * m_maxNeighbors + n]], m_h);
                }            
            }
        }
    }       
    return (scalar)(m_fpmass) * sumGrad / m_p0; 

}

void SerialFluid::calculateLambdas(){
    memset(m_lambda, 0, getNumParticles() * sizeof(scalar)); 

    scalar top = 0; 
    scalar gradSum; 
    scalar gradL; 
    int gi; 
    for(int p = 0; p < getNumParticles(); ++p){
        gradSum = 0;
        top = -(m_pcalc[p]/m_p0 - 1.0); 
        // for all neighbors, calculate Constraint gradient at p 
        for(int i = std::max(0, m_gridInd[p*3]-1); i <= std::min(m_gridX-1, m_gridInd[p*3]+1); ++i){
            for(int j = std::max(0, m_gridInd[p*3+1]-1); j <= std::min(m_gridY-1, m_gridInd[p*3+1]+1); ++j){
                for(int k = std::max(0, m_gridInd[p*3+2]-1); k <= std::min(m_gridZ-1, m_gridInd[p*3+2]+1); ++k){
                    gi = getGridIdx(i, j, k); // current grid
                    for(int n = 0; n < m_gridCount[gi]; ++n){ // for all particles in the grid
                        gradL =  glm::length(calcGradConstraint(m_ppos[p], m_ppos[m_grid[gi * m_maxNeighbors + n]]));  
                        gradSum += gradL * gradL;
                    }            
                }
            }
        }        
        
        // add self-gradient 
        gradL = glm::length(calcGradConstraintAtI(p)); 
        gradSum += gradL * gradL; 
        m_lambda[p] = top / (gradSum + m_eps); 
    }
}

// Calculate appropriate change in position to best solve constraint
void SerialFluid::calculatedPos(){
    memset(m_dpos, 0, getNumParticles() * sizeof(Vector3s)); 
    for(int p = 0; p < getNumParticles(); ++p){
        Vector3s dp(0.0, 0.0, 0.0); 

        int q = 0;
        int gi = 0; 
        scalar scorr = 0; // actually calculate
        // for all neighbors, calculate Constraint gradient at p 
        for(int i = std::max(0, m_gridInd[p*3]-1); i <= std::min(m_gridX-1, m_gridInd[p*3]+1); ++i){
            for(int j = std::max(0, m_gridInd[p*3+1]-1); j <= std::min(m_gridY-1, m_gridInd[p*3+1]+1); ++j){
                for(int k = std::max(0, m_gridInd[p*3+2]-1); k <= std::min(m_gridZ-1, m_gridInd[p*3+2]+1); ++k){
                    gi = getGridIdx(i, j, k); // current grid
                    for(int n = 0; n < m_gridCount[gi]; ++n){ // for all particles in the grid
                        q = m_grid[gi * m_maxNeighbors + n]; 
                        dp += (m_lambda[p] + m_lambda[q] + scorr)*wSpikyKernelGrad(m_ppos[p], m_ppos[q], m_h);            

                    }            
                }
            }
        }        
        m_dpos[p] = (scalar)(m_fpmass) * dp / m_p0;    
    }
}


void SerialFluid::accumulateForce(Scene& scene){
    std::vector<FluidForce*> fluidForces = scene.getFluidForces();

    // init F to 0 
    memset (m_accumForce, 0, getNumParticles() * sizeof(Vector3s)); 
    for(int i = 0; i < fluidForces.size(); ++i){
        fluidForces[i]->addGradEToTotal(m_pos, m_vel, m_fpmass, m_accumForce, getNumParticles());
    }

    for(int i = 0; i < getNumParticles(); ++i){
        m_accumForce[i] /= -m_fpmass; 
    }
}

void SerialFluid::updateVelocityFromForce(scalar dt){
    for(int i = 0; i < getNumParticles(); ++i){
        m_vel[i] += m_accumForce[i] * dt; 
    }    
}

void SerialFluid::updatePredPosition(scalar dt){
    for(int i = 0; i < getNumParticles(); ++i){
        m_ppos[i] = m_pos[i] + m_vel[i] * dt;
    }
}

// Determine which grid this position is in
void SerialFluid::getGridIdx(Vector3s &pos, int &idx, int &idy, int &idz){
    idx = (pos[0] - m_boundingBox->minX())/m_h; 
    idy = (pos[1] - m_boundingBox->minY())/m_h;
    idz = (pos[2] - m_boundingBox->minZ())/m_h; 

    assert(idx >= 0);
    assert(idy >= 0);
    assert(idz >= 0);
    assert(idx < m_gridX);
    assert(idy < m_gridY);
    assert(idz < m_gridZ);
}

int SerialFluid::getGridIdx(int i, int j, int k){
    return (m_gridX * m_gridY) * k + (m_gridX) * j + i; 
}

void SerialFluid::clearGrid(){
    memset(m_grid, -1, m_gridX * m_gridY * m_gridZ * m_maxNeighbors * sizeof(int));
    memset(m_gridCount, 0,  m_gridX * m_gridY * m_gridZ * sizeof(int)); 
}


// Each particle calculates its index in the grid
// The grid gets its list of particles... 
void SerialFluid::buildGrid(){
    for(int i= 0; i < getNumParticles(); ++i){
        getGridIdx(m_ppos[i], m_gridInd[i*3], m_gridInd[i*3+1], m_gridInd[i*3+2]); // which grid location am I in 
     }

    // zero things out
    clearGrid(); 

    // Build list of grid particles
    int gind; 
    for(int i = 0; i < getNumParticles(); ++i){
        gind = getGridIdx(m_gridInd[i*3], m_gridInd[i*3+1], m_gridInd[i*3+2]); 
        if(m_gridCount[gind] < m_maxNeighbors){
            m_grid[gind * m_maxNeighbors + m_gridCount[gind]] = i; 
            m_gridCount[gind] ++; 
        }
    }

}

void SerialFluid::preserveOwnBoundary(){
    m_boundingBox->dealWithCollisions(m_ppos, m_dpos, getNumParticles());
}

void SerialFluid::dealWithCollisions(Scene& scene){
    std::vector<FluidBoundary *>bounds = scene.getFluidBoundaries(); 
    for(int i = 0; i < bounds.size(); ++i){
        bounds[i]->dealWithCollisions(m_ppos, m_dpos, getNumParticles()); 
    }
}

void SerialFluid::recalculateVelocity(scalar dt){
    for(int i = 0; i < getNumParticles(); ++i){
        m_vel[i] = (m_ppos[i] - m_pos[i])/dt; 
    }
}

void SerialFluid::updateFinalPosition(){
    Vector3s *temp = m_pos; 
    m_pos = m_ppos; // predicted positions become real positions
    m_ppos = temp; // recalculate predicted positions anyway
}

void SerialFluid::applydPToPredPos(){
    for(int i = 0; i < getNumParticles(); ++i){
        m_ppos[i] += m_dpos[i]; 
    }
    
}

void SerialFluid::updateVBO(float* dptrvert)
{
}
