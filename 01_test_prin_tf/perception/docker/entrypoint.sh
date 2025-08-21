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
  echo "Building bringup_ws..."
  cd /root/bringup_ws
  colcon build --symlink-install
  source install/setup.bash
else
  echo "Skipping bringup_ws - directory not found"
fi

if [ -d "/opt/third_party/livox_ws" ]; then
  echo "Building livox_ws..."
  cd /opt/third_party/livox_ws
  if [ -f "install/setup.bash" ]; then
    source install/setup.bash
    echo "Livox workspace setup completed"
  else
    echo "pwd: $(pwd)"
    echo "⚠️  Warning: install/setup.bash not found in livox_ws - skipping setup"
  fi
else
  echo "Skipping livox_ws - directory not found"
fi

echo "ROS2 workspace setup completed successfully!"

# Execute the command passed to the container
exec "$@"
