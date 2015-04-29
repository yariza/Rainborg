#include "SerialFluid.h"
#include "Scene.h"

SerialFluid::SerialFluid(scalar mass, scalar p0, scalar h, int iters, int maxNeighbors, int minNeighbors)
: Fluid(mass, p0, h, iters, maxNeighbors, minNeighbors)
, m_eps(.01) // wow this is terrible
{
    // Allocate memory for m_pos, m_ppos, m_vel, m_accumForce? 

    // DEFER ALL LOADING TO Fluid.loadFluidVolumes()

    // m_numParticles = numParticles;
    // //m_pos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    // //m_ppos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    // m_lambda = (scalar *)malloc(numParticles * sizeof(scalar)); 
    // m_pcalc = (scalar *)malloc(numParticles * sizeof(scalar)); 
    // memset(m_pcalc, 0, numParticles * sizeof(scalar));
    // //m_dpos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    // //m_vel = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    // //m_accumForce = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    
    // m_pos = (Vector3s *)malloc(numParticles * sizeof(Vector3s));
    // m_ppos = (Vector3s *)malloc(numParticles * sizeof(Vector3s)); 
    // m_dpos = (Vector3s *)malloc(numParticles * sizeof(Vector3s));
    // m_vel = (Vector3s *)malloc(numParticles * sizeof(Vector3s)); 
    // m_accumForce = (Vector3s *)malloc(numParticles * sizeof(Vector3s)); 
    // m_colors = (Vector4s *)malloc(numParticles * sizeof(Vector4s)); 

    // m_gridInd = (int *)malloc(numParticles * 3 * sizeof(int));
    m_grid = NULL; // don't know bounding box yet
    m_gridCount = NULL;

    // assert (m_pos != NULL);
    // assert(m_ppos != NULL);
    // assert(m_lambda != NULL); 
    // assert(m_pcalc != NULL); 
    // assert(m_dpos != NULL);
    // assert(m_vel != NULL);
    // assert(m_gridInd != NULL);
    // assert(m_accumForce != NULL);
    // assert(m_colors != NULL); 
}


SerialFluid::SerialFluid(const SerialFluid& otherFluid)
: Fluid(otherFluid)
, m_eps(.01)
{

    // loadFluidVolumes is necessary to reallocate stuff again

    // Allocate memory 
    //m_pos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    //m_ppos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    // m_lambda = (scalar *)malloc(m_numParticles * sizeof(scalar));
    // m_pcalc = (scalar *)malloc(m_numParticles * sizeof(scalar)); 
    // memset(m_pcalc, 0, m_numParticles * sizeof(scalar));
    //m_dpos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar)); 
    //m_vel = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    //m_accumForce = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    // m_gridInd = (int *)malloc(m_numParticles * 3 * sizeof(int));

    // m_pos = (Vector3s *)malloc(m_numParticles * sizeof(Vector3s));
    // m_ppos = (Vector3s *)malloc(m_numParticles * sizeof(Vector3s)); 
    // m_dpos = (Vector3s *)malloc(m_numParticles * sizeof(Vector3s));
    // m_vel = (Vector3s *)malloc(m_numParticles * sizeof(Vector3s)); 
    // m_accumForce = (Vector3s *)malloc(m_numParticles * sizeof(Vector3s)); 
    // m_colors = (Vector4s *)malloc(m_numParticles * sizeof(Vector4s)); 

    // assert (m_pos != NULL);
    // assert(m_ppos != NULL);
    // assert(m_pcalc != NULL); 
    // assert(m_lambda != NULL); 
    // assert(m_dpos != NULL);
    // assert(m_vel != NULL);
    // assert(m_accumForce != NULL);
    // assert(m_gridInd != NULL); 
    // assert(m_colors != NULL); 
    // Set positions, velocity 
    // Note: predicted positions, accumulatedForces are recalculated each time step so no point copying those


    //memcpy(m_pos, otherFluid.getFPPos(), m_numParticles * 3 * sizeof(scalar));
    //memcpy(m_vel, otherFluid.getFPVel(), m_numParticles * 3 * sizeof(scalar));

    // memcpy(m_pos, otherFluid.getFPPos(), m_numParticles * sizeof(Vector3s)); 
    // memcpy(m_vel, otherFluid.getFPVel(), m_numParticles * sizeof(Vector3s)); 
    // memcpy(m_colors, otherFluid.getColors(), m_numParticles * sizeof(Vector4s)); 

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


// use fluid volumes instead
// void SerialFluid::setFPPos(int fp, const Vector3s& pos){
//     // assert(fp >= 0 && fp < m_numParticles);

//     m_pos[fp] = pos; 

//     //    m_pos[fp*3] = pos[0];
//     //    m_pos[fp*3+1] = pos[1];
//     //    m_pos[fp*3+2] = pos[2];
// }

void SerialFluid::setFPVel(int fp, const Vector3s& vel){
    // assert(fp >= 0 && fp < m_numParticles);
    m_vel[fp] = vel;  

    //    m_vel[fp*3] = vel[0];
    //    m_vel[fp*3+1] = vel[1];
    //    m_vel[fp*3+2] = vel[2];
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

//scalar* SerialFluid::getFPPos() const{
//    return m_pos;
//}

//scalar* SerialFluid::getFPVel() const{
//    return m_vel;
//}

Vector3s* SerialFluid::getFPPos() const{
    return m_pos; 
}

Vector3s* SerialFluid::getFPVel() const{
    return m_vel;
}
/*
Vector4s* SerialFluid::getColors() const{
    return m_colors; 
}
*/
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

void SerialFluid::stepSystem(Scene& scene, scalar dt){
    //std::cout << "step" << std::endl;

    accumulateForce(scene); // makes more sense 

    //printVec3(m_vel[0]); 
    updateVelocityFromForce(dt); 
    
    updatePredPosition(dt); 
    
    // make sure that predicted positions don't go out of bounds here 
    memset(m_dpos, 0, getNumParticles() * sizeof(Vector3s)); 
    preserveOwnBoundary(); 
    
    applydPToPredPos();
    //std::cout << "building grid" << std::endl;  
    buildGrid();   // Or at least, since neighbors are just adjacent grids, build grid structure

    // loop for solve iterations
    for(int loop = 0; loop < m_iters; ++loop){
        //std::cout << "in loop " << loop << std::endl;
        // calculate lambda for each particle
        //std::cout << "calculating pressures" << std::endl;
        calculatePressures(); 
        //std::cout << "calculating lambdas" << std::endl;
        calculateLambdas();  
        // Calculate dpos for each particle
        //std::cout << "calculating dpos" << std::endl;
        calculatedPos(); 

        // Deal with collision detection and response
        //std::cout << "dealing with collisions" << std::endl;
        dealWithCollisions(scene); 
        //std::cout << "own boundary" << std::endl;
        preserveOwnBoundary(); 

        // Update predicted position with dP
        //std::cout << "applying dpos to ppos" << std::endl;
        applydPToPredPos(); 
    }
    //std::cout << "end loop" << std::endl;

    // Update velocities
    //std::cout << "recalculating velocities" << std::endl;
    recalculateVelocity(dt); 
    //std::cout << "new vel: " << std::endl;
    //updateVelocityFromForce(dt);  // Yeah no this shouldn't be here

    // Apply vorticity confinement and XSPH viscosity

    //std::cout << "updating final positions" << std::endl;
    updateFinalPosition(); 

    //std::cout << "end step system" << std::endl;
}

// Wow this is the ugliest loop something is sure to be wrong somewhere
// If not enough neighbors? Chosen behaviour: set pressure to rest; don't act on constraints
void SerialFluid::calculatePressures(){
    //std::cout << "starting Pressure calculations" << std::endl;
    int gi = 0; // current grid id
    scalar press; 
    int ncount; // number of neighbors
    for(int p = 0; p < getNumParticles(); ++p){
        // grab neighbors?  
        ncount = 0; 
        press = 0;
        for(int i = std::max(0, m_gridInd[p*3]-1); i <= std::min(m_gridX-1, m_gridInd[p*3]+1); ++i){
            for(int j = std::max(0, m_gridInd[p*3+1]-1); j <= std::min(m_gridY-1, m_gridInd[p*3+1]+1); ++j){
                for(int k = std::max(0, m_gridInd[p*3+2]-1); k <= std::min(m_gridZ-1, m_gridInd[p*3+2]+1); ++k){
                    gi = getGridIdx(i, j, k); 
                    //std::cout << "grid " << gi << " has " << m_gridCount[gi] << std::endl;
                    for(int n = 0; n < m_gridCount[gi]; ++n){ // for all particles in the grid
                        //std::cout << "m_grid[gi]: " << m_grid[gi] << std::endl;
                        //std::cout << "neighbor at: " << m_grid[gi *m_maxNeighbors + n] << std::endl;
                        //printVec3(m_ppos[p]); 
                        //printVec3(m_ppos[m_grid[gi] * m_maxNeighbors + n]);
                        scalar pressN = wPoly6Kernel(m_ppos[p], m_ppos[m_grid[gi * m_maxNeighbors + n]], m_h); 
                        press += pressN;
                        //press += wPoly6Kernel(m_ppos[p], m_ppos[m_grid[gi * m_maxNeighbors + n]], m_h); 
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
        
        //std::cout << "particle " << p << " has " << ncount << "neighbors" << std::endl;
        
        /*
        if(p == 700){
            std::cout << "arb count: " << ncount << std::endl;
            std::cout << "arb press: " << m_fpmass * press << std::endl;
        }
        */

    }
    //std::cout << "ending Pressure calculations" << std::endl;
}

Vector3s SerialFluid::calcGradConstraint(Vector3s& pi, Vector3s& pj){
    return (scalar)(m_fpmass) * wSpikyKernelGrad(pi, pj, m_h) / (- m_p0); 
}

Vector3s SerialFluid::calcGradConstraintAtI(int p){
    // Bah
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

    // F *= -1.0/mass
    for(int i = 0; i < getNumParticles(); ++i){
        m_accumForce[i] /= -m_fpmass; 
        //m_accumForce[i*3] /= -m_fpmass; 
        //m_accumForce[i*3+1] /= -m_fpmass;
        //m_accumForce[i*3+2] /= -m_fpmass;
    }
    //printVec3(m_accumForce[0]); 
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

void SerialFluid::getGridIdx(Vector3s &pos, int &idx, int &idy, int &idz){
    // in our case...
    //printVec3(pos);

    idx = (pos[0] - m_boundingBox->minX())/m_h; 
    idy = (pos[1] - m_boundingBox->minY())/m_h;
    idz = (pos[2] - m_boundingBox->minZ())/m_h; 

    assert(idx >= 0);
    assert(idy >= 0);
    assert(idz >= 0);
    assert(idx < m_gridX);
    assert(idy < m_gridY);
    assert(idz < m_gridZ);
    //idx = (m_gridX * m_gridY) * k + (m_gridX) * j + i; 
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
// GPU version presumably with atomic adds? hopefully particles won't try to write to the 
// same grid at once too often, but when it does, deal with it atomically? 
// Still sucks for coalescence and things though. 
void SerialFluid::buildGrid(){
    for(int i= 0; i < getNumParticles(); ++i){
        //getGridIdx(m_ppos[i*3], m_ppos[i*3+1], m_ppos[i*3+2], m_gridInd[i]); 
        getGridIdx(m_ppos[i], m_gridInd[i*3], m_gridInd[i*3+1], m_gridInd[i*3+2]); // which grid location am I in 
     }

    // zero things out
    clearGrid(); 

    // Build list of grid particles
    int gind; 
    for(int i = 0; i < getNumParticles(); ++i){
        gind = getGridIdx(m_gridInd[i*3], m_gridInd[i*3+1], m_gridInd[i*3+2]); 
        m_grid[gind * m_maxNeighbors + m_gridCount[gind]] = i; 
        m_gridCount[gind] ++; 
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
        
        if(glm::length(m_dpos[i]) > 5.0){
            std::cout << "huge update at " << i << std::endl;
            std::cout << "  pressure: " << m_pcalc[i] << std::endl;
            std::cout << "  lambda: " << m_lambda[i] << std::endl;
            std::cout << "  grid: " << m_gridInd[i] << std::endl;
            std::cout << "  grid count: " << m_gridCount[m_gridInd[i]] << std::endl;

        }
        
    }
}

void SerialFluid::updateVBO(float* dptrvert)
{
}
