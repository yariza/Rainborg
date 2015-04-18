#include "Fluid.h"
#include "Scene.h"

Fluid::Fluid(int numParticles, scalar mass, scalar p0, scalar h, int iters, int maxNeighbors) 
: m_fpmass(mass)
, m_p0(p0)
, m_h(h)
, m_iters(iters)
, m_maxNeighbors(maxNeighbors)
{
    // Allocate memory for m_pos, m_ppos, m_vel, m_accumForce? 

    m_numParticles = numParticles;
    m_pos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_ppos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_lambda = (scalar *)malloc(numParticles * sizeof(scalar)); 
    m_dpos = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_vel = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_accumForce = (scalar *)malloc(numParticles * 3 * sizeof(scalar));
    m_gridInd = (int *)malloc(numParticles * sizeof(int));
    m_grid = NULL; // don't know bounding box yet
    m_gridCount = NULL;

    assert (m_pos != NULL);
    assert(m_ppos != NULL);
    assert(m_lambda != NULL); 
    assert(m_dpos != NULL);
    assert(m_vel != NULL);
    assert(m_gridInd != NULL);
    assert(m_accumForce != NULL);
}


Fluid::Fluid(const Fluid& otherFluid){
    m_numParticles = otherFluid.getNumParticles();
    m_fpmass = otherFluid.getFPMass();
    m_p0 = otherFluid.getRestDensity();
    m_h = otherFluid.getKernelH(); 
    m_iters = otherFluid.getNumIterations(); 
    m_maxNeighbors = otherFluid.getMaxNeighbors();

    // Allocate memory 
    m_pos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_ppos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_lambda = (scalar *)malloc(m_numParticles * sizeof(scalar)); 
    m_dpos = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar)); 
    m_vel = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_accumForce = (scalar *)malloc(m_numParticles * 3 * sizeof(scalar));
    m_gridInd = (int *)malloc(m_numParticles * sizeof(int));
 

    assert (m_pos != NULL);
    assert(m_ppos != NULL);
    assert(m_lambda != NULL); 
    assert(m_dpos != NULL);
    assert(m_vel != NULL);
    assert(m_accumForce != NULL);
    assert(m_gridInd != NULL); 
    // Set positions, velocity 
    // Note: predicted positions, accumulatedForces are recalculated each time step so no point copying those


    memcpy(m_pos, otherFluid.getFPPos(), m_numParticles * 3 * sizeof(scalar));
    memcpy(m_vel, otherFluid.getFPVel(), m_numParticles * 3 * sizeof(scalar));


    m_boundingBox = otherFluid.getBoundingBox(); 

    m_gridX = ceil(m_boundingBox.width()/m_h);
    m_gridY = ceil(m_boundingBox.height()/m_h);
    m_gridZ = ceil(m_boundingBox.depth()/m_h);


    m_grid = (int *)malloc(m_gridX * m_gridY * m_gridZ * m_maxNeighbors * sizeof(int)); 
    m_gridCount = (int *)malloc(m_gridX * m_gridY * m_gridZ * sizeof(int)); 

   
    
    assert(m_grid != NULL);
    assert(m_gridCount != NULL);
}

Fluid::~Fluid(){
    free(m_pos);
    free(m_ppos);
    free(m_lambda);
    free(m_dpos);
    free(m_vel);
    free(m_accumForce);
    free(m_gridInd);
    if(m_grid != NULL)
        free(m_grid);
    if(m_gridCount != NULL)
        free(m_gridCount);
}

void Fluid::setFPMass(scalar fpm){
    assert(fpm > 0); 
    m_fpmass = fpm;
}

void Fluid::setRestDensity(scalar p0){
    m_p0 = p0;
}

void Fluid::setFPPos(int fp, const Vector3s& pos){
    assert(fp >= 0 && fp < m_numParticles);
    
    m_pos[fp*3] = pos[0];
    m_pos[fp*3+1] = pos[1];
    m_pos[fp*3+2] = pos[2];
}

void Fluid::setFPVel(int fp, const Vector3s& vel){
    assert(fp >= 0 && fp < m_numParticles);
    
    m_vel[fp*3] = vel[0];
    m_vel[fp*3+1] = vel[1];
    m_vel[fp*3+2] = vel[2];
}

void Fluid::setKernelH(scalar h){
    m_h = h; 
}

void Fluid::setNumIterations(int iter){
    m_iters = iter;
}

void Fluid::setBoundingBox(FluidBoundingBox& bound){
    m_boundingBox = bound; 

    if(m_grid != NULL)
        free(m_grid);
    if(m_gridCount != NULL)
        free(m_gridCount);

    m_gridX = ceil(m_boundingBox.width()/m_h);
    m_gridY = ceil(m_boundingBox.height()/m_h);
    m_gridZ = ceil(m_boundingBox.depth()/m_h);
    m_grid = (int *)malloc(m_gridX * m_gridY * m_gridZ * m_maxNeighbors * sizeof(int)); 
    m_gridCount = (int *)malloc(m_gridX * m_gridY * m_gridZ * sizeof(int)); 

    assert(m_grid != NULL);
    assert(m_gridCount != NULL);
}

int Fluid::getMaxNeighbors() const{
    return m_maxNeighbors;
}

int Fluid::getNumParticles() const{
    return m_numParticles;
}

scalar Fluid::getFPMass() const{
    return m_fpmass;
}

scalar Fluid::getRestDensity() const{
    return m_p0;
}
 
scalar Fluid::getKernelH() const{
    return m_h;
}

int Fluid::getNumIterations() const{
    return m_iters;
}

scalar* Fluid::getFPPos() const{
    return m_pos;
}

scalar* Fluid::getFPVel() const{
    return m_vel;
}


const FluidBoundingBox& Fluid::getBoundingBox() const{
    return m_boundingBox;
}


void Fluid::stepSystem(Scene& scene, scalar dt){
    accumulateForce(scene); // makes more sense 
    updateVelocityFromForce(dt); 
    updatePredPosition(dt); 

    // find neighbors for each particle 
    buildGrid();   // Or at least, since neighbors are just adjacent grids, build grid structure

    // loop for solve iterations
    for(int loop = 0; loop < m_iters; ++loop){
        // calculate lambda for each particle
        
        // Calculate dpos for each particle
    
        // Deal with collision detection and response
        dealWithCollisions(scene); 
        preserveOwnBoundary(); 
    

        // Update predicted position with dP
        applydPToPredPos(); 
    }
    
    // Update velocities
    recalculateVelocity(dt); 
    // Apply vorticity confinement and XSPH viscosity

    updateFinalPosition(); 
    
}

void Fluid::accumulateForce(Scene& scene){
    std::vector<FluidForce*> fluidForces = scene.getFluidForces();

    // init F to 0 
    memset (m_accumForce, 0, m_numParticles * 3 * sizeof(scalar)); 
    for(int i = 0; i < fluidForces.size(); ++i){
        fluidForces[i]->addGradEToTotal(m_pos, m_vel, m_fpmass, m_accumForce, m_numParticles);
    }
    
    // F *= -1.0/mass
    for(int i = 0; i < m_numParticles; ++i){
        m_accumForce[i*3] /= -m_fpmass; 
        m_accumForce[i*3+1] /= -m_fpmass;
        m_accumForce[i*3+2] /= -m_fpmass;
    }
}

void Fluid::updateVelocityFromForce(scalar dt){
    for(int i = 0; i < m_numParticles*3; ++i){
        m_vel[i] += m_accumForce[i] * dt; 
    }    
}

void Fluid::updatePredPosition(scalar dt){
    for(int i = 0; i < m_numParticles*3; ++i){
        m_ppos[i] = m_pos[i] + m_vel[i] * dt;
    }
}

void Fluid::getGridIdx(scalar x, scalar y, scalar z, int& idx){
    // in our case...
    int i = (x - m_boundingBox.minX())/m_h; 
    int j = (y - m_boundingBox.minY())/m_h;
    int k = (z - m_boundingBox.minZ())/m_h; 

    idx = (m_gridX * m_gridY) * k + (m_gridX) * j + i; 

}

void Fluid::clearGrid(){
    memset(m_grid, -1, m_gridX * m_gridY * m_gridZ * m_maxNeighbors * sizeof(int));
    memset(m_gridCount, 0,  m_gridX * m_gridY * m_gridZ * sizeof(int)); 

//    std::cout << m_boundingBox.minX() << ": " << m_boundingBox.maxX() << std::endl;
//    std::cout << m_h << std::endl;
//    std::cout << m_gridX * m_gridY * m_gridZ << std::endl;
//    std::cout << m_grid[2] << std::endl;
//    std::cout << m_gridCount[3] << std::endl;
}


// Each particle calculates its index in the grid
// The grid gets its list of particles... 
// GPU version presumably with atomic adds? hopefully particles won't try to write to the 
// same grid at once too often, but when it does, deal with it atomically? 
// Still sucks for coalescence and things though. 
void Fluid::buildGrid(){
    for(int i= 0; i < m_numParticles; ++i){
        getGridIdx(m_ppos[i*3], m_ppos[i*3+1], m_ppos[i*3+2], m_gridInd[i]); 
    }

    // zero things out
    clearGrid(); 

    // Build list of grid particles
    int gind; 
    for(int i = 0; i < m_numParticles; ++i){
        gind = m_gridInd[i];
        m_grid[gind * m_maxNeighbors + m_gridCount[gind]] = i; 
        m_gridCount[gind] ++; 
    }
}

void Fluid::preserveOwnBoundary(){
    m_boundingBox.dealWithCollisions(m_ppos, m_dpos, m_numParticles);
}

void Fluid::dealWithCollisions(Scene& scene){


}

void Fluid::recalculateVelocity(scalar dt){
    for(int i = 0; i < m_numParticles*3; ++i){
        m_vel[i] = (m_ppos[i] - m_pos[i])/dt; 
    }
}

void Fluid::updateFinalPosition(){
    scalar *temp = m_pos; 
    m_pos = m_ppos; // predicted positions become real positions
    m_ppos = temp; // recalculate predicted positions anyway
}

void Fluid::applydPToPredPos(){
    for(int i = 0; i < m_numParticles * 3; ++i){
        m_ppos[i] += m_dpos[i]; 
    }
}

