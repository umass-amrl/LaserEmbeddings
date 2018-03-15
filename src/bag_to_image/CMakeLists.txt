SET(target BagToImages)
ROSBUILD_ADD_EXECUTABLE(${target}
   bag_to_images.cpp)

ADD_DEPENDENCIES(${target}
   shared_libraries
   perception_library
   rosbuild_precompile)

TARGET_LINK_LIBRARIES(${target}
   shared_libraries
   perception_library
   ${libs}
   gflags)

SET(target AnalyzeScans)
ROSBUILD_ADD_EXECUTABLE(${target}
   analyze_scans.cpp)

ADD_DEPENDENCIES(${target}
   shared_libraries
   perception_library
   gui_libraries
   rosbuild_precompile)

TARGET_LINK_LIBRARIES(${target}
   shared_libraries
   perception_library
   ${libs}
   gflags
   gui_libraries)
