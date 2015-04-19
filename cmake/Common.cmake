# don't build in the source directory
if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
  message (SEND_ERROR "Do not build in the source directory.")
  message (FATAL_ERROR "Remove the created \"CMakeCache.txt\" file and the \"CMakeFiles\" directory, then create a build directory and call \"${CMAKE_COMMAND} <path to the sources>\".")
endif ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")

# finds all files with a given extension
macro (append_files files ext)
  foreach (dir ${ARGN})
    file (GLOB _files "${CMAKE_CURRENT_LIST_DIR}/${dir}/*.${ext}")
    list (APPEND ${files} ${_files})
  endforeach (dir)
endmacro (append_files)

macro(copy_files TGT_NAME GLOBPAT DESTINATION)
  file(GLOB COPY_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLOBPAT})

  foreach(FILENAME ${COPY_FILES})
    get_filename_component(BASENAME ${FILENAME} NAME)
    set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
    set(DST "${DESTINATION}/${BASENAME}")

    add_custom_command(
      OUTPUT ${DST}
      COMMAND ${CMAKE_COMMAND} -E copy ${SRC} ${DST}
      COMMENT "Copying ${SRC} to ${DST}"
      MAIN_DEPENDENCY ${SRC}
      )
    set(COPY_TGT_NAME "copy.${TGT_NAME}.${BASENAME}")
    add_custom_target(
        ${COPY_TGT_NAME}
        DEPENDS ${DST}
    )
    add_dependencies(${TGT_NAME} ${COPY_TGT_NAME})
  endforeach(FILENAME)
endmacro(copy_files)
