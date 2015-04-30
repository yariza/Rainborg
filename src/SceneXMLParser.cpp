#include "SceneXMLParser.h"
#include "StringUtilities.h"
#include "MathDefs.h"
#include "FluidSimpleGravityForce.h"
#include "GridGPUFluid.h"
#include "NaiveGPUFluid.h"

void SceneXMLParser::loadSimulation(const std::string& file_name,
                                    bool rendering_enabled,
                                    bool gpu_enabled,
                                    openglframework::GLFWViewer* viewer,
                                    Simulation** sim,
                                    scalar& dt,
                                    scalar& max_time,
                                    scalar& steps_per_sec_cap,
                                    openglframework::Color& bgcolor,
                                    std::string& description) {

    // Load the xml document
    std::vector<char> xmlchars;
    rapidxml::xml_document<> doc;
    loadXMLFile( file_name, xmlchars, doc );

    // Attempt to locate the root node
    rapidxml::xml_node<>* node = doc.first_node("scene");
    if( node == NULL )
    {
      std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse xml scene file. Failed to locate root <scene> node. Exiting." << std::endl;
      exit(1);
    }

    loadMaxTime(node, max_time);
    loadMaxSimFrequency(node, steps_per_sec_cap);
    loadCamera(node, viewer);
    loadViewport(node, viewer);

    Stepper* stepper = NULL;
    loadStepper(node, dt, &stepper);
    loadBackgroundColor(node, bgcolor);
    loadSceneDescriptionString(node, description);

    Scene* scene = new Scene();

    loadSimpleGravityForces(node, *scene);

    loadFluids(node, *scene, gpu_enabled);

    SceneRenderer *renderer = NULL;
    if (rendering_enabled)
        renderer = new SceneRenderer(scene);

    *sim = new Simulation(scene, stepper, renderer);
}

void SceneXMLParser::loadMaxTime(rapidxml::xml_node<>* node, scalar& max_t) {

    assert(node != NULL);

    // Attempt to locate the duraiton node
    rapidxml::xml_node<>* nd = node->first_node("duration");
    if( nd == NULL )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "No duration specified. Exiting." << std::endl;
        exit(1);
    }

    // Attempt to load the duration value
    rapidxml::xml_attribute<>* timend = nd->first_attribute("time");
    if( timend == NULL )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "No duration 'time' attribute specified. Exiting." << std::endl;
        exit(1);
    }

    max_t = std::numeric_limits<scalar>::signaling_NaN();
    if( !stringutils::extractFromString(std::string(timend->value()),max_t) )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse 'time' attribute for duration. Value must be numeric. Exiting." << std::endl;
      exit(1);
    }
}

void SceneXMLParser::loadMaxSimFrequency( rapidxml::xml_node<>* node, scalar& max_freq ) {

    assert( node != NULL );

    // Attempt to locate the duraiton node
    if( node->first_node("maxsimfreq") )
    {
        // Attempt to load the duration value
        rapidxml::xml_attribute<>* atrbnde = node->first_node("maxsimfreq")->first_attribute("max");
        if( atrbnde == NULL )
        {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                      << "No maxsimfreq 'max' attribute specified. Exiting." << std::endl;
            exit(1);
        }

        if( !stringutils::extractFromString(std::string(atrbnde->value()),max_freq) )
        {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                      << "Failed to parse 'max' attribute for maxsimfreq. Value must be scalar. Exiting." << std::endl;
                exit(1);
        }
    }
}

void SceneXMLParser::loadViewport(rapidxml::xml_node<> *node, openglframework::GLFWViewer* viewer) {

    assert( node != NULL );

    float width = 500, height = 500;

    rapidxml::xml_node<>* nd = node->first_node("viewport");
    if (nd) {
        if (nd->first_attribute("width")) {
            std::string attribute(nd->first_attribute("width")->value());
            if (!stringutils::extractFromString(attribute, width)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of width attribute for viewport. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        if (nd->first_attribute("height")) {
            std::string attribute(nd->first_attribute("height")->value());
            if (!stringutils::extractFromString(attribute, height)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of height attribute for viewport. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
    }
    else {
        std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "Viewport not specified. Using default values of " << width << " by " << height << "." << std::endl;
    }
    viewer->setWindowSize(width, height);
}

void SceneXMLParser::loadCamera(rapidxml::xml_node<> *node, openglframework::GLFWViewer* viewer) {
    assert(node != NULL);

    openglframework::Camera &camera = viewer->getCamera();

    float cx = 0, cy = 0, cz = 0;
    float near = 0.1, far = 10.0;
    float fov = 45;
    float rotx = 0, roty = 0, rotz = 0;
    bool cameraRotationSpecified = true;

    rapidxml::xml_node<>* nd = node->first_node("camera");
    if (nd) {
        if (nd->first_attribute("cx")) {
            std::string attribute(nd->first_attribute("cx")->value());
            if (!stringutils::extractFromString(attribute, cx)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of cx attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                      << "cx attribute for camera not specified. Using default value of " << cx << "." << std::endl;
        }

        if (nd->first_attribute("cy")) {
            std::string attribute(nd->first_attribute("cy")->value());
            if (!stringutils::extractFromString(attribute, cy)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of cy attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                      << "cy attribute for camera not specified. Using default value of " << cy << "." << std::endl;
        }

        if (nd->first_attribute("cz")) {
            std::string attribute(nd->first_attribute("cz")->value());
            if (!stringutils::extractFromString(attribute, cz)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of cz attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                      << "cz attribute for camera not specified. Using default value of " << cz << "." << std::endl;
        }

        if (nd->first_attribute("near")) {
            std::string attribute(nd->first_attribute("near")->value());
            if (!stringutils::extractFromString(attribute, near)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of near attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }

        if (nd->first_attribute("far")) {
            std::string attribute(nd->first_attribute("far")->value());
            if (!stringutils::extractFromString(attribute, far)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of far attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }

        if (nd->first_attribute("fov")) {
            std::string attribute(nd->first_attribute("fov")->value());
            if (!stringutils::extractFromString(attribute, fov)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of fov attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }

        if (nd->first_attribute("rotx")) {
            std::string attribute(nd->first_attribute("rotx")->value());
            if (!stringutils::extractFromString(attribute, rotx)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of rotx attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                      << "rotx attribute for camera not specified. Using default values." << std::endl;
            cameraRotationSpecified = false;
        }

        if (nd->first_attribute("roty")) {
            std::string attribute(nd->first_attribute("roty")->value());
            if (!stringutils::extractFromString(attribute, roty)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of roty attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                      << "roty attribute for camera not specified. Using default values." << std::endl;
            cameraRotationSpecified = false;
        }

        if (nd->first_attribute("rotz")) {
            std::string attribute(nd->first_attribute("rotz")->value());
            if (!stringutils::extractFromString(attribute, rotz)) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                          << "Failed to parse value of rotz attribute for camera. Value must be numeric. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                      << "rotz attribute for camera not specified. Using default values." << std::endl;
            cameraRotationSpecified = false;
        }
    }
    else {
        std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "Camera not specified. Using default values." << std::endl;
    }

    if (cameraRotationSpecified) {
        camera.translateWorld( openglframework::Vector3(cx, cy, cz));
        camera.rotateWorld(openglframework::Vector3(1,0,0), rotx * DEG_TO_RAD);
        camera.rotateWorld(openglframework::Vector3(0,1,0), roty * DEG_TO_RAD);
        camera.rotateWorld(openglframework::Vector3(0,0,1), rotz * DEG_TO_RAD);
        camera.setClippingPlanes(near, far);
        camera.setFieldOfView(fov);
    }
    else {
        float radius = 10.0f;
        openglframework::Vector3 center(0.0, 0.0, 0.0);

        viewer->setScenePosition(center, radius);
    }
}

void SceneXMLParser::loadStepper(rapidxml::xml_node<>* node, scalar& dt, Stepper** stepper) {

    assert(node != NULL);

    dt = -1.0;

    rapidxml::xml_node<>* nd = node->first_node("stepper");
    if( nd == NULL )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "No stepper specified. Exiting." << std::endl;
        exit(1);
    }

    // Attempt to load the duration value
    rapidxml::xml_attribute<>* timend = nd->first_attribute("dt");
    if( timend == NULL )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "No stepper 'dt' attribute specified. Exiting." << std::endl;
        exit(1);
    }

    dt = std::numeric_limits<scalar>::signaling_NaN();
    if( !stringutils::extractFromString(std::string(timend->value()),dt) )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse 'dt' attribute for stepper. Value must be numeric. Exiting." << std::endl;
        exit(1);
    }

    *stepper = new Stepper();
}

void SceneXMLParser::loadBackgroundColor( rapidxml::xml_node<>* node, openglframework::Color& color )
{
  if( rapidxml::xml_node<>* nd = node->first_node("backgroundcolor") )
  {
    // Read in the red color channel
    double red = -1.0;
    if( nd->first_attribute("r") )
    {
      std::string attribute(nd->first_attribute("r")->value());
      if( !stringutils::extractFromString(attribute,red) )
      {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of r attribute for backgroundcolor. Value must be scalar. Exiting." << std::endl;
        exit(1);
      }
    }
    else
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of r attribute for backgroundcolor. Exiting." << std::endl;
      exit(1);
    }

    if( red < 0.0 || red > 1.0 )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of r attribute for backgroundcolor. Invalid color specified. Valid range is " << 0.0 << "..." << 1.0 << std::endl;
      exit(1);
    }


    // Read in the green color channel
    double green = -1.0;
    if( nd->first_attribute("g") )
    {
      std::string attribute(nd->first_attribute("g")->value());
      if( !stringutils::extractFromString(attribute,green) )
      {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of g attribute for backgroundcolor. Value must be scalar. Exiting." << std::endl;
        exit(1);
      }
    }
    else
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of g attribute for backgroundcolor. Exiting." << std::endl;
      exit(1);
    }

    if( green < 0.0 || green > 1.0 )
    {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of g attribute for backgroundcolor. Invalid color specified. Valid range is " << 0.0 << "..." << 1.0 << std::endl;
      exit(1);
    }


    // Read in the blue color channel
    double blue = -1.0;
    if( nd->first_attribute("b") )
    {
      std::string attribute(nd->first_attribute("b")->value());
      if( !stringutils::extractFromString(attribute,blue) )
      {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of b attribute for backgroundcolor. Value must be scalar. Exiting." << std::endl;
        exit(1);
      }
    }
    else
    {
      std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of b attribute for backgroundcolor. Exiting." << std::endl;
      exit(1);
    }

    if( blue < 0.0 || blue > 1.0 )
    {
      std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
            << "Failed to parse value of b attribute for backgroundcolor. Invalid color specified. Valid range is " << 0.0 << "..." << 1.0 << std::endl;
      exit(1);
    }

    //std::cout << red << "   " << green << "   " << blue << std::endl;

    color.r = red;
    color.g = green;
    color.b = blue;
  }
}

void SceneXMLParser::loadSceneDescriptionString( rapidxml::xml_node<>* node, std::string& description_string )
{
    assert( node != NULL );

    description_string = "No description specified.";

    // Attempt to locate the integrator node
    rapidxml::xml_node<>* nd = node->first_node("description");
    if( nd != NULL )
    {
        rapidxml::xml_attribute<>* typend = nd->first_attribute("text");
        if( typend == NULL )
        {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                    << "No text attribute specified for description. Exiting." << std::endl;
            exit(1);
        }
        description_string = typend->value();
    }
}

void SceneXMLParser::loadSimpleGravityForces(rapidxml::xml_node<>* node, Scene& scene) {

    assert(node != NULL);

    int forcenum = 0;
    for (rapidxml::xml_node<>* nd = node->first_node("fluidsimplegravityforce"); nd; nd = nd->next_sibling("fluidsimplegravityforce")) {

        scalar fx, fy, fz;

        if (nd->first_attribute("fx")) {
            std::string attribute(nd->first_attribute("fx")->value());
            if( !stringutils::extractFromString(attribute,fx) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse fx attribute for simple gravity force. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing fx attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("fy")) {
            std::string attribute(nd->first_attribute("fy")->value());
            if( !stringutils::extractFromString(attribute,fy) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse fy attribute for simple gravity force. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing fy attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("fz")) {
            std::string attribute(nd->first_attribute("fz")->value());
            if( !stringutils::extractFromString(attribute,fz) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse fz attribute for simple gravity force. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing fz attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        FluidSimpleGravityForce* force = new FluidSimpleGravityForce(Vector3s(fx, fy, fz));

        scene.insertFluidForce(force);
        forcenum++;
    }
}

void SceneXMLParser::loadFluids(rapidxml::xml_node<>* node, Scene& scene, bool gpu_enabled) {

    assert(node != NULL);

    int fluidsnum = 0;
    for (rapidxml::xml_node<>* nd = node->first_node("fluid"); nd; nd = nd->next_sibling("fluid")) {

        int numParticles = 0;
        scalar mass, p0, h, iters;
        int minneighbors, maxneighbors;
        std::string type;

        if (nd->first_attribute("mass")) {
            std::string attribute(nd->first_attribute("mass")->value());
            if( !stringutils::extractFromString(attribute,mass) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of mass attribute for fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing mass attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("p0")) {
            std::string attribute(nd->first_attribute("p0")->value());
            if( !stringutils::extractFromString(attribute,p0) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of p0 attribute for fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing p0 attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("h")) {
            std::string attribute(nd->first_attribute("h")->value());
            if( !stringutils::extractFromString(attribute,h) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of h attribute for fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing h attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("iters")) {
            std::string attribute(nd->first_attribute("iters")->value());
            if( !stringutils::extractFromString(attribute,iters) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of iters attribute for fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing iters attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("minneighbors")) {
            std::string attribute(nd->first_attribute("minneighbors")->value());
            if( !stringutils::extractFromString(attribute,minneighbors) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of minneighbors attribute for fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing minneighbors attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("maxneighbors")) {
            std::string attribute(nd->first_attribute("maxneighbors")->value());
            if( !stringutils::extractFromString(attribute,maxneighbors) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of maxneighbors attribute for fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing maxneighbors attribute for fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("type")) {
            type = (nd->first_attribute("type")->value());
            if( type != "naive" && type != "grid" ) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of type attribute for fluid. Value must be either naive or grid. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            if (gpu_enabled) {
                type = "grid";
            }
        }

        Fluid *fluid = NULL;
        if (gpu_enabled) {
            if (type == "grid") {
                fluid = new GridGPUFluid(mass, p0, h, iters, maxneighbors, minneighbors);
            }
            else {
                fluid = new NaiveGPUFluid(mass, p0, h, iters, maxneighbors, minneighbors);
            }
        }
        else {
            fluid = new SerialFluid(mass, p0, h, iters, maxneighbors, minneighbors);
        }

        loadFluidBoundingBox(nd, *fluid);
        loadFluidVolumes(nd, *fluid);

        scene.insertFluid(fluid);
        fluidsnum++;
    }

    if (fluidsnum == 0) {
        std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "No fluids in scene." << std::endl;
    }
}

void SceneXMLParser::loadFluidBoundingBox(rapidxml::xml_node<>* node, Fluid& fluid) {

    assert(node != NULL);

    rapidxml::xml_node<>* nd = node->first_node("boundingbox");
    if (nd) {
        scalar xmin, xmax, ymin, ymax, zmin, zmax;

        if (nd->first_attribute("xmin")) {
            std::string attribute(nd->first_attribute("xmin")->value());
            if( !stringutils::extractFromString(attribute,xmin) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of xmin attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing xmin attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("xmax")) {
            std::string attribute(nd->first_attribute("xmax")->value());
            if( !stringutils::extractFromString(attribute,xmax) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of xmax attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing xmax attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("ymin")) {
            std::string attribute(nd->first_attribute("ymin")->value());
            if( !stringutils::extractFromString(attribute,ymin) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of ymin attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing ymin attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("ymax")) {
            std::string attribute(nd->first_attribute("ymax")->value());
            if( !stringutils::extractFromString(attribute,ymax) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of ymax attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing ymax attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("zmin")) {
            std::string attribute(nd->first_attribute("zmin")->value());
            if( !stringutils::extractFromString(attribute,zmin) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of zmin attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing zmin attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("zmax")) {
            std::string attribute(nd->first_attribute("zmax")->value());
            if( !stringutils::extractFromString(attribute,zmax) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of zmax attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing zmax attribute for bounding box of fluid. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        FluidBoundingBox* boundingbox = new FluidBoundingBox(xmin, xmax, ymin, ymax, zmin, zmax);

        fluid.setBoundingBox(boundingbox);
    }
    else {
        std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
              << "Missing bounding box for fluid. Exiting." << std::endl;
        exit(1);
    }
}

void SceneXMLParser::loadFluidVolumes(rapidxml::xml_node<>* node, Fluid& fluid) {

    assert(node != NULL);

    int volumenum = 0;
    for (rapidxml::xml_node<>* nd = node->first_node("fluidvolume"); nd; nd = nd->next_sibling("fluidvolume")) {

        scalar xmin, xmax, ymin, ymax, zmin, zmax;
        scalar r = 0.317, g = 0.639, b = 1.0, a = 0.5;
        int numparticles = 100;
        fluid_volume_mode_t mode;
        bool random;
        scalar spacing;
        bool particle_selected = false;
        bool spacing_selected = false;

        if (nd->first_attribute("xmin")) {
            std::string attribute(nd->first_attribute("xmin")->value());
            if( !stringutils::extractFromString(attribute,xmin) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of xmin attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing xmin attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("xmax")) {
            std::string attribute(nd->first_attribute("xmax")->value());
            if( !stringutils::extractFromString(attribute,xmax) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of xmax attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing xmax attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("ymin")) {
            std::string attribute(nd->first_attribute("ymin")->value());
            if( !stringutils::extractFromString(attribute,ymin) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of ymin attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing ymin attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("ymax")) {
            std::string attribute(nd->first_attribute("ymax")->value());
            if( !stringutils::extractFromString(attribute,ymax) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of ymax attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing ymax attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("zmin")) {
            std::string attribute(nd->first_attribute("zmin")->value());
            if( !stringutils::extractFromString(attribute,zmin) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of zmin attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing zmin attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("zmax")) {
            std::string attribute(nd->first_attribute("zmax")->value());
            if( !stringutils::extractFromString(attribute,zmax) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of zmax attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing zmax attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("numparticles")) {
            std::string attribute(nd->first_attribute("numparticles")->value());
            if( !stringutils::extractFromString(attribute,numparticles) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of numparticles attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
            particle_selected = true;
        }
        if (nd->first_attribute("spacing")) {
            std::string attribute(nd->first_attribute("spacing")->value());
            if( !stringutils::extractFromString(attribute,spacing) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of spacing attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
            spacing_selected = true;
        }
        if (!particle_selected && !spacing_selected) {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing either spacing or numparticles attribute for fluid volume. Values must be scalar. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("mode")) {
            std::string mode_string(nd->first_attribute("mode")->value());
            if( mode_string != "box" && mode_string != "sphere" )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of mode attribute for fluid volume. Value must be either box or sphere. Exiting." << std::endl;
              exit(1);
            }
            if (mode_string == "box") {
                mode = kFLUID_VOLUME_MODE_BOX;
            }
            else if (mode_string == "sphere") {
                mode = kFLUID_VOLUME_MODE_SPHERE;
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing mode attribute for fluid volume. Value must be either box or sphere. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("random")) {
            std::string attribute(nd->first_attribute("random")->value());
            if( !stringutils::extractBoolFromString(attribute,random) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of random attribute for fluid volume. Value must be boolean. Exiting." << std::endl;
              exit(1);
            }
        }
        else {
            std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Missing random attribute for fluid volume. Value must be boolean. Exiting." << std::endl;
            exit(1);
        }

        if (nd->first_attribute("r")) {
            std::string attribute(nd->first_attribute("r")->value());
            if( !stringutils::extractFromString(attribute,r) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of r attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
            if (r < 0 || r > 1) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of r attribute for fluid volume. Value must be between 0 and 1. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "Missing r attribute for fluid volume. Using default values." << std::endl;
        }

        if (nd->first_attribute("g")) {
            std::string attribute(nd->first_attribute("g")->value());
            if( !stringutils::extractFromString(attribute,g) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of g attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
            if (g < 0 || g > 1) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of g attribute for fluid volume. Value must be between 0 and 1. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "Missing g attribute for fluid volume. Using default values." << std::endl;
        }

        if (nd->first_attribute("b")) {
            std::string attribute(nd->first_attribute("b")->value());
            if( !stringutils::extractFromString(attribute,b) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of b attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
            if (b < 0 || b > 1) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of b attribute for fluid volume. Value must be between 0 and 1. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "Missing b attribute for fluid volume. Using default values." << std::endl;
        }

        if (nd->first_attribute("a")) {
            std::string attribute(nd->first_attribute("a")->value());
            if( !stringutils::extractFromString(attribute,a) )
            {
              std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of a attribute for fluid volume. Value must be scalar. Exiting." << std::endl;
              exit(1);
            }
            if (a < 0 || a > 1) {
                std::cerr << outputmod::startred << "ERROR IN XMLSCENEPARSER:" << outputmod::endred
                  << "Failed to parse value of a attribute for fluid volume. Value must be between 0 and 1. Exiting." << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "Missing a attribute for fluid volume. Using default values." << std::endl;
        }

        FluidVolume volume(xmin, xmax, ymin, ymax, zmin, zmax, numparticles, mode, random,
                           r, g, b, a);
        if (spacing_selected) {
            volume.setSpacing(spacing);
        }
        fluid.insertFluidVolume(volume);

        volumenum++;
    }

    if (volumenum == 0) {
        std::cerr << outputmod::startpink << "Warning in XMLSceneParser:" << outputmod::endpink
                  << "No fluid volumes in fluid." << std::endl;
    }
}

void SceneXMLParser::loadXMLFile( const std::string& filename, std::vector<char>& xmlchars, rapidxml::xml_document<>& doc )
{
  // Attempt to read the text from the user-specified xml file
  std::string filecontents;
  if( !loadTextFileIntoString(filename,filecontents) )
  {
    std::cerr << "\033[31;1mERROR IN TWODSCENEXMLPARSER:\033[m XML scene file " << filename << ". Failed to read file." << std::endl;
    exit(1);
  }

  // Copy string into an array of characters for the xml parser
  for( int i = 0; i < (int) filecontents.size(); ++i ) xmlchars.push_back(filecontents[i]);
  xmlchars.push_back('\0');

  // Initialize the xml parser with the character vector
  doc.parse<0>(&xmlchars[0]);
}

bool SceneXMLParser::loadTextFileIntoString( const std::string& filename, std::string& filecontents )
{
  // Attempt to open the text file for reading
  std::ifstream textfile(filename.c_str(),std::ifstream::in);
  if(!textfile) return false;

  // Read the entire file into a single string
  std::string line;
  while(getline(textfile,line)) filecontents.append(line);

  textfile.close();

  return true;
}

