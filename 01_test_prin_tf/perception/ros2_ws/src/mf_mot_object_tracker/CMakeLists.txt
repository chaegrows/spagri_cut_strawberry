cmake_minimum_required(VERSION 3.8)
project(mf_mot_object_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

set(EXECUTABLE_DIR scripts)

install(PROGRAMS
  # ${EXECUTABLE_DIR}/mot_by_bbox.py
  # ${EXECUTABLE_DIR}/mot_by_bbox_server.py
  # ${EXECUTABLE_DIR}/mot_by_bbox_pre_a.py
  ${EXECUTABLE_DIR}/params/mot_by_bbox/strawberry_harvesting_pollination.py
  ${EXECUTABLE_DIR}/yolo.py
  # ${EXECUTABLE_DIR}/yolo_multi.py
  #${EXECUTABLE_DIR}/test_keypoints.py
  ${EXECUTABLE_DIR}/001_generate_keypoints.py
  #${EXECUTABLE_DIR}/002_generate_tf.py
  #${EXECUTABLE_DIR}/003_generate_tf_from_keypoints.py
  # ${EXECUTABLE_DIR}/yolo_keypoints.py
  #${EXECUTABLE_DIR}/test_keypoints_acc.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
