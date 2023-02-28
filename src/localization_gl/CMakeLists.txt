cmake_minimum_required(VERSION 3.16)
project(localization_gl)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
    message("OpenGL found")
    message ("OPENGL_LIBRARIES: " ${OPENGL_LIBRARIES})
endif()

find_package(GLUT REQUIRED)
if(GLUT_FOUND)
    message("GLUT found")
    message ("GLUT_glut_LIBRARY: " ${GLUT_glut_LIBRARY})
endif()

find_package(CUDA 10.0 REQUIRED)
if(CUDA_FOUND)
    message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ;-gencode arch=compute_86,code=sm_86; -Xptxas -v;--expt-extended-lambda;-std=c++14)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ;-allow-unsupported-compiler)
include_directories(3rd/imgui 3rd/imgui/backends)

include_directories(3rd/imgui 3rd/imgui/backends include)
set(ImGui_Glut_src 3rd/imgui/imgui.cpp 3rd/imgui/imgui_demo.cpp 3rd/imgui/imgui_draw.cpp 3rd/imgui/imgui_widgets.cpp
        3rd/imgui/imgui_tables.cpp
        3rd/imgui/backends/imgui_impl_glut.cpp 3rd/imgui/backends/imgui_impl_opengl2.cpp)

if(CATKIN_DEVEL_PREFIX OR catkin_FOUND OR CATKIN_BUILD_BINARY_PACKAGE)
  message(STATUS "Localization GL for ROS1 will be compiled")

  find_package(catkin REQUIRED COMPONENTS pcl_ros roscpp sensor_msgs tf2 tf2_ros)

  catkin_package()

  include_directories(
    ${catkin_INCLUDE_DIRS}
  )

  cuda_add_executable(localization_gl_node src/localization_gl_node.cpp ${ImGui_Glut_src} src/pf.cpp src/rgd.cu src/cuda_exception.cpp)
  cuda_add_executable(test_trajectory src/test_trajectory.cpp ${ImGui_Glut_src} src/pf.cpp src/rgd.cu src/cuda_exception.cpp)
  cuda_add_executable(test_cov src/test_cov.cpp ${ImGui_Glut_src} src/pf.cpp src/rgd.cu src/cuda_exception.cpp)
  cuda_add_executable(testFrame src/test.cpp ${ImGui_Glut_src} src/pf.cpp src/rgd.cu src/cuda_exception.cpp)

  target_link_libraries(test_trajectory
          ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})
  target_link_libraries(test_cov
          ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})
  target_link_libraries(testFrame
          ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})

  ## Specify libraries to link a library or executable target against
  target_link_libraries(localization_gl_node
    ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY}
  )
elseif(DEFINED ENV{AMENT_PREFIX_PATH})
  message(STATUS "Localization GL for ROS2 will be compiled")

  find_package (Eigen3 3.3 NO_MODULE)
  find_package(ament_cmake REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(pcl_ros REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(PCL REQUIRED)

  include_directories(
          include
          ${PCL_INCLUDE_DIRS}
          ${rclcpp_INCLUDE_DIRS})

  cuda_add_executable(localization_gl_node src/localization_gl_node_ros2.cpp ${ImGui_Glut_src} src/pf.cpp src/rgd.cu src/cuda_exception.cpp)
  cuda_add_executable(save_pointcloud_node src/save_pointcloud_ros2.cpp)

  cuda_add_executable(testFrame src/test.cpp ${ImGui_Glut_src} src/pf.cpp src/rgd.cu src/cuda_exception.cpp)


  target_link_libraries(testFrame
          ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} Eigen3::Eigen ${PCL_LIBRARIES})
  ## Specify libraries to link a library or executable target against
  target_link_libraries(localization_gl_node
    ${catkin_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} Eigen3::Eigen
  )
  ament_target_dependencies(localization_gl_node rclcpp std_msgs nav_msgs sensor_msgs tf2_ros pcl_ros)
  ament_target_dependencies(save_pointcloud_node rclcpp std_msgs nav_msgs sensor_msgs tf2_ros pcl_ros)

else()
  message(FATAL_ERROR "catkin or colcon not found Localization GL disabled")
endif()