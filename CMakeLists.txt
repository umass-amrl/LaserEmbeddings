PROJECT(Neural_Laser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(arch ${})

MESSAGE(STATUS "Compiling Human-in-the-Loop sources...")
MESSAGE(STATUS "Compilers found: ${CMAKE_CXX_COMPILER_LIST}")
MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CXX_FLAGS "-std=c++11 -march=native -Wall -g -rdynamic")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  MESSAGE(STATUS "Additional Flags for Release mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -O3 -DNDEBUG")
  SET(BUILD_SPECIFIC_LIBRARIES "tcmalloc")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  MESSAGE(STATUS "Additional Flags for Debug mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -rdynamic")
  SET(BUILD_SPECIFIC_LIBRARIES "")
ENDIF()

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init()
SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "Build-Specific Libraries: ${BUILD_SPECIFIC_LIBRARIES}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(QT_USE_QTNETWORK true)
SET(QT_USE_QTOPENGL true)

#===============================================================================
# Targets selection
OPTION(TESTER_TARGETS "Enable compilation of tester targets" FALSE)
OPTION(RESEARCH_TARGETS "Enable compilation of research targets" FALSE)
OPTION(UTIL_TARGETS "Enable compilation of research targets" FALSE)
OPTION(CALIBRATION_TARGETS "Enable complication of delta-calibration" FALSE)
#===============================================================================

#===============================================================================
# Ceres-Solver Library, Xml2
IF (RESEARCH_TARGETS)

  FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

  FIND_PACKAGE(LibXml2 REQUIRED)
  INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})

  FIND_PACKAGE(OpenCV REQUIRED)

ENDIF (RESEARCH_TARGETS)
#===============================================================================

FIND_PACKAGE(freenect2 REQUIRED)
INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
FIND_PACKAGE(Qt4 REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
#FIND_PACKAGE(USB REQUIRED)
FIND_PACKAGE(Lua51 REQUIRED)
FIND_PACKAGE(SDL REQUIRED)
FIND_PACKAGE(SDL_ttf REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

SET(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
   "/usr/share/cmake/Modules")
#FIND_PACKAGE(pocketsphinx REQUIRED) --UNCOMMENT BEFORE PUSH
#INCLUDE_DIRECTORIES(${POCKETSPHINX_INCLUDE_DIRS}) --UNCOMMENT BEFORE PUSH

SET(LIBQR_LIBRARIES /usr/local/lib/libdecodeqr.so)

INCLUDE(${QT_USE_FILE})

FIND_PATH(GLEW_INCLUDE_PATH GL/glew.h
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    DOC "The directory where GL/glew.h resides")

FIND_LIBRARY(GLEW_LIBRARY
  NAMES GLEW glew
  PATHS
  /usr/lib64
  /usr/lib
  /usr/local/lib64
  /usr/local/lib
  /sw/lib
  /opt/local/lib
  DOC "The GLEW library")


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/perception_tools)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/shared/math)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/shared/util)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/bag_to_image)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/perception_tools)

INCLUDE_DIRECTORIES(${GTK2_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${USB_INCLUDE_DIR})

SET(libs roslib roscpp ncurses popt pthread ${GTK2_LIBRARIES} glut
    ${OPENGL_LIBRARY} ${GLEW_LIBRARY} ${LUA_LIBRARIES} rt
    ${BUILD_SPECIFIC_LIBRARIES} glog rosbag X11)


SET(SHARED_SRCS
  src/shared/math/gvector.h
  src/shared/math/line.h
  src/shared/math/gvector.cpp
  src/shared/math/geomalgo.h
  src/shared/math/geometry.h
  src/shared/util/helpers.cpp
  src/shared/util/proghelp.cc
  src/shared/util/pthread_utils.cpp
  src/shared/util/terminal_utils.cpp
  src/shared/util/configreader.cpp
  src/shared/util/watch_files.cpp
)

SET(PERCEPTION_SRCS
  src/perception_tools/perception_2d.cpp
)

#================================================================================
#Creating libraries to eliminate redundant compile steps

ADD_LIBRARY(shared_libraries SHARED ${SHARED_SRCS})
ADD_DEPENDENCIES(shared_libraries rosbuild_precompile)
TARGET_LINK_LIBRARIES(shared_libraries shared_libraries ${libs})

ADD_LIBRARY(perception_library ${PERCEPTION_SRCS})
ADD_DEPENDENCIES(perception_library shared_libraries)
TARGET_LINK_LIBRARIES(perception_library shared_libraries ${libs})
