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

echo "ROS2 workspace setup completed successfully!"

# Execute the command passed to the container
exec "$@"
