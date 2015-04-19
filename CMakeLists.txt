# Minimum cmake version required
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Project configuration
PROJECT(pbf)

# Add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Add cmake file with useful functions
include (Common)

# Add source files
append_files( Headers "h" src/ )
append_files( Sources "cpp" src/ )

# Find OpenGL
find_package( OpenGL REQUIRED )
if( OPENGL_FOUND)
    include_directories(${OpenGL_INCLUDE_DIRS})
    set( pbf_LIBRARIES ${pbf_LIBRARIES} ${OPENGL_LIBRARIES})
else ( OPENGL_FOUND )
    message(ERROR " OPENGL not found!")
endif( OPENGL_FOUND)

# TCLAP library is required
find_package (TCLAP REQUIRED)
if (TCLAP_FOUND)
  include_directories (${TCLAP_INCLUDE_PATH})
else (TCLAP_FOUND)
  message (SEND_ERROR "Unable to locate TCLAP")
endif (TCLAP_FOUND)

# RapidXML library is required
find_package (RapidXML REQUIRED)
if (RAPIDXML_FOUND)
  include_directories (${RAPIDXML_INCLUDE_DIR})
else (RAPIDXML_FOUND)
  message (SEND_ERROR "Unable to locate RapidXML")
endif (RAPIDXML_FOUND)

# GLM library
find_package( GLM REQUIRED )
if ( GLM_FOUND )
    include_directories(${GLM_INCLUDE_DIRS})
else( GLM_FOUND )
    message(ERROR " GLM not found")
endif( GLM_FOUND )

add_subdirectory(openglframework)
include_directories(${CMAKE_SOURCE_DIR}/include/openglframework)

# copy shaders
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/shaders/")

add_executable ( testflow ${Headers} ${Templates} ${Sources})


target_link_libraries( testflow ${EXTRA_LIBS} ${pbf_LIBRARIES} openglframework )

