#ifndef __SCENE_XML_PARSER_H__
#define __SCENE_XML_PARSER_H__

#include <iostream>
#include <fstream>
#include "rapidxml.hpp"

#include <openglframework.h>
#include "Simulation.h"

class SceneXMLParser {

public:
    void loadSimulation(const std::string& file_name,
                        bool rendering_enabled,
                        openglframework::GLFWViewer* viewer,
                        Simulation** sim,
                        scalar& dt,
                        scalar& max_time,
                        scalar& steps_per_sec_cap,
                        openglframework::Color& bgcolor,
                        std::string& description);


    void loadXMLFile( const std::string& filename, std::vector<char>& xmlchars, rapidxml::xml_document<>& doc );
    void loadTextFileIntoString( const std::string& filename, std::string& filecontents );
};

#endif