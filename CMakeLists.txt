cmake_minimum_required(VERSION 3.8)
project(e2enav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find libtorch
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../imitation_nav/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find other dependencies
find_package(OpenCV REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Create executable directly (no separate library needed)
ament_auto_add_executable(inference_node
  src/inference_node.cpp
)

target_link_libraries(inference_node
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

target_include_directories(inference_node PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Install
install(DIRECTORY
  include/
  DESTINATION include/
)

install(DIRECTORY
  launch
  config
  scripts
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS 
  inference_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()