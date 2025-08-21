cmake_minimum_required(VERSION 3.5)

project(mf_msgs)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)


set(msg_files
  "msg/BehaviorTreeCommand.msg"
  "msg/BehaviorTreeStatus.msg"
  "msg/Heartbeat.msg"
  "msg/HealthManager.msg"
  "msg/Endeffector.msg"
  "msg/LiftControl.msg"
  "msg/LiftStatus.msg"
  "msg/MD200Control.msg"
  "msg/MD200Status.msg"
  "msg/ManipulatorControl.msg"
  "msg/ManipulatorStatus.msg"
)

set(srv_files
  "srv/MfService.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
  # ADD_LINTER_TESTS
)
ament_export_dependencies(rosidl_default_runtime)


ament_package()
