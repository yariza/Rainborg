# Minimum cmake version required
cmake_minimum_required(VERSION 2.6)

# Project configuration
PROJECT(OPENGLFRAMEWORKDEMO)

# Headers
INCLUDE_DIRECTORIES(${OPENGLFRAMEWORKDEMO_SOURCE_DIR})

# Create the demo executable
ADD_EXECUTABLE(demo main.cpp Scene.h Scene.cpp)

TARGET_LINK_LIBRARIES(demo openglframework)

# copy resources
copy_files(demo "torus.obj" ${CMAKE_CURRENT_BINARY_DIR})
copy_files(demo "../src/shaders/*" ${CMAKE_CURRENT_BINARY_DIR}/shaders)
