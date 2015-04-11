# Rainborg
Parallelized ray tracer capable of rendering rainbows

### External Libraries
    OpenGL
    GLM
    GLFW
    RapidXML
    TCLAP

### Todo: (actually this time)

- Rendering
    + Spheres (with lighting)
    + Optional import mesh, boundary
- Simulation
    + Do the thing
    + Go through FOSSSim, stepper
    + Boundary/collision detection handling (static rigid objects)
    + Serial
        * external forces (gravity)
        * predict position
        * find neighbors
        * jacobi iterator
            - constraint, lambda
            - s-correction
            - calculate delta-p
            - collision detection response
            - update positions
        * vorticity confinement
        * position update
    + Parallel
        * Buckets, arbitrary acceleration data structures
        * same thing...
    + Future

### References 

- [Physically-based Simulation of Rainbows](http://graphics.ucsd.edu/~henrik/papers/physically_based_simulation_of_rainbows.pdf)
- [Rendering Light Dispersion with a Composite Spectral Model](https://www.cs.sfu.ca/~mark/ftp/Cgip00/dispersion_CGIP00.pdf)
- [Computing the scattering properties of participating media using Lorenz-Mie theory](http://dl.acm.org/citation.cfm?id=1276452)
- [Rainbow Tutorial - The Great Skywatcher](http://darksilverflame.deviantart.com/art/Rainbow-Tutorial-The-Great-Skywatcher-Guide-201667461)
- [GPU Papers](https://mediatech.aalto.fi/~timo/HPG2009/index.html)
- [Voxels](http://research.michael-schwarz.com/publ/files/vox-siga10.pdf)
- [GPU Voxels](http://research.michael-schwarz.com/publ/files/vox-siga10.pdf)

### Todo:

- Look into SFML
- Triangle mesh representation (design for easy bucket-ing to voxels)
- Parsing (probably xml unless we find something better)
- UI
    + camera controls (WASD-FG) (if we get it working semi-realtime)
    + extra: dragging camera
- Initial prototype:
    + AO
- Look into dispersion, refraction, transparent materials oh god (BTDF)
- Write math classes for Vector operations etc. (no Eigen :( )
- Figure out what exactly to parallelize (e.g. voxellization, monte carlo integration, channel-specific)
- Look into cuRAND/other random number generators
- Triangle rasterization onto voxels: GPU problem
