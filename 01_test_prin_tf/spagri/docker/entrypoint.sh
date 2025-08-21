#!/bin/bash

echo "Starting ROS2 workspace setup..."

# Source ROS2 environment
echo "Setting up ROS2 environment..."
source /opt/ros/humble/setup.bash

# Build and source ros2_ws
echo "Building ros2_ws..."
cd /root/ros2_ws
colcon build --symlink-install
source install/setup.bash

# Build and source bringup_ws if it exists
if [ -d "/root/bringup_ws" ]; then
  # if [ -d "/root/bringup_ws/src/frcobot_ros2" ]; then
    # mkdir -p /root/bringup_ws/src
    # cd /root/bringup_ws/src
    # wget https://raw.githubusercontent.com/ros-controls/ros2_control_ci/master/ros_controls.$ROS_DISTRO.repos
    # vcs import src < ros_controls.$ROS_DISTRO.repos
    # rosdep update --rosdistro=$ROS_DISTRO
    # apt-get update
    # rosdep install --from-paths src --ignore-src -r -y
    # apt-get install ros-humble-moveit*
  # fi
  echo "Building bringup_ws..."
  cd /root/bringup_ws
  # colcon build --symlink-install
  source install/setup.bash
else
  echo "Skipping bringup_ws - directory not found"
fi


# if [ -d "/opt/third_party/livox_ws" ]; then
#   echo "Building livox_ws..."
#   cd /opt/third_party/livox_ws
#   if [ -f "install/setup.bash" ]; then
#     source install/setup.bash
#     echo "Livox workspace setup completed"
#   else
#     echo "pwd: $(pwd)"
#     echo "⚠️  Warning: install/setup.bash not found in livox_ws - skipping setup"
#   fi
# else
#   echo "Skipping livox_ws - directory not found"
# fi


echo "ROS2 workspace setup completed successfully!"

# Execute the command passed to the container
exec "$@"
