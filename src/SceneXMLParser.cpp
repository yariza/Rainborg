#include "SceneXMLParser.h"

void SceneXMLParser::loadSimulation(const std::string& file_name,
                        bool rendering_enabled,
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
      std::cerr << "\033[31;1mERROR IN XMLSCENEPARSER:\033[m Failed to parse xml scene file. Failed to locate root <scene> node. Exiting." << std::endl;
      exit(1);
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

