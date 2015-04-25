# Minimum cmake version required
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Project configuration
PROJECT(pbf)

option(GPU_ENABLED "Compile with GPU support" ON)

# Add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Add cmake file with useful functions
include (Common)

# Add source files
append_files( Headers "h" src/ /src/gpu/)
append_files( Sources "cpp" src/ /src/gpu/)

# Find OpenGL
find_package( OpenGL REQUIRED )
if( OPENGL_FOUND)
    set( pdf_INCLUDE_DIRS ${pdf_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIRS})
    set( pbf_LIBRARIES ${pbf_LIBRARIES} ${OPENGL_LIBRARIES})
else ( OPENGL_FOUND )
    message(ERROR " OPENGL not found!")
endif( OPENGL_FOUND)

# TCLAP library is required
find_package (TCLAP REQUIRED)
if (TCLAP_FOUND)
  set( pdf_INCLUDE_DIRS ${pdf_INCLUDE_DIRS} ${TCLAP_INCLUDE_PATH})
else (TCLAP_FOUND)
  message (SEND_ERROR "Unable to locate TCLAP")
endif (TCLAP_FOUND)

# RapidXML library is required
find_package (RapidXML REQUIRED)
if (RAPIDXML_FOUND)
  set( pdf_INCLUDE_DIRS ${pdf_INCLUDE_DIRS} ${RAPIDXML_INCLUDE_DIR})
else (RAPIDXML_FOUND)
  message (SEND_ERROR "Unable to locate RapidXML")
endif (RAPIDXML_FOUND)

# GLM library
find_package( GLM REQUIRED )
if ( GLM_FOUND )
  set( pdf_INCLUDE_DIRS ${pdf_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS})
else( GLM_FOUND )
    message(ERROR " GLM not found")
endif( GLM_FOUND )

option (USE_PNG "Builds in support for exporting to png (UNSUPPORTED)" OFF)
if (USE_PNG)
  find_package (PNG)
  if (PNG_FOUND)
    add_definitions (-DPNGOUT)
    add_definitions (${PNG_DEFINITIONS})
    set( pdf_INCLUDE_DIRS ${pdf_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})
    set (pbf_LIBRARIES ${pbf_LIBRARIES} ${PNG_LIBRARIES})
  endif (PNG_FOUND)
endif (USE_PNG)

add_subdirectory(openglframework)
set( pdf_INCLUDE_DIRS ${pdf_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include/openglframework)

if (GPU_ENABLED)
    find_package(CUDA)
    # nvcc flags
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_20,code=sm_20)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20)
    append_files(CudaFiles "cu" /src /src/gpu)
    #cuda_compile(CUDA_O ${CudaFiles})
    # add def for program to check
    add_definitions(-DGPU_ENABLED)
endif (GPU_ENABLED)

if (GPU_ENABLED)
   cuda_include_directories(${pdf_INCLUDE_DIRS})
   cuda_add_executable(testflow ${Headers} ${Templates} ${Sources} ${CudaFiles})
else (GPU_ENABLED)
   add_executable ( testflow ${Headers} ${Templates} ${Sources} ${CUDA_O})
endif (GPU_ENABLED)

# copy shaders
copy_files(testflow "src/shaders/*" ${CMAKE_CURRENT_BINARY_DIR}/shaders)


target_link_libraries( testflow ${EXTRA_LIBS} ${pbf_LIBRARIES} openglframework)

