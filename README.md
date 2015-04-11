# Rainborg
Parallelized ray tracer capable of rendering rainbows

### External Libraries
- OpenGL
- GLM
- GLFW
- RapidXML
- TCLAP

### Structure
- Main: 
    + Parse command line arguments
    + Windows and display loop
    + Initialize things?? 
        - Load Files
        - Create scenes + data structures
- Scene: 
    + Boundaries -> Mesh objects
    + FluidSim
        - Fluid Particles
            - mass, rest density (shared)
            - Position
            - Velocity
            - Color
        - Bounding box
        - FluidSim.step(dt, scene)
    + Forces (gravity)

       

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

