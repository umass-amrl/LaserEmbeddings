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

