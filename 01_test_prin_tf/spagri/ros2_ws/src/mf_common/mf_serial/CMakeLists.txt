cmake_minimum_required(VERSION 3.10)

project(mf_serial VERSION 1.0 LANGUAGES CXX)
# add_compile_options(-std=c++17 -Wall -Wextra -Werror)
add_compile_options(-std=c++17 -g -Wall -Wextra)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mf_msgs REQUIRED)
find_package(serial_driver REQUIRED)
find_package(yaml-cpp REQUIRED)



include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${serial_driver_INCLUDE_DIRS}
)

add_executable(mf_eef_driver
  src/mf_eef_driver.cpp
  src/mf_serial_base.cpp)

ament_target_dependencies(mf_eef_driver
  serial_driver
  mf_msgs
  yaml-cpp
)
target_link_libraries(mf_eef_driver yaml-cpp)


add_executable(lift_main
  src/lift_main.cpp
  src/mf_serial_base.cpp)

ament_target_dependencies(lift_main
  serial_driver
  mf_msgs
  yaml-cpp
)

target_link_libraries(lift_main yaml-cpp)

add_executable(md_main
  src/md_main.cpp
  src/mf_serial_base.cpp)

ament_target_dependencies(md_main
  serial_driver
  mf_msgs
  yaml-cpp
)

target_link_libraries(md_main yaml-cpp)


install(TARGETS
  lift_main
  md_main
  mf_eef_driver
  DESTINATION lib/${PROJECT_NAME})

ament_package()

