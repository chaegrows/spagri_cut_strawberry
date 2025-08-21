#!/bin/bash

set -e

source /opt/ros/humble/setup.bash

# Create workspace structure
mkdir -p /root/ros2_ws/src
mkdir -p /root/bringup_ws/src

# Copy selected bringup_ws packages
IFS=',' read -ra WS_LIST <<< "$BRINGUP_WS_LIST"
for ws in "${WS_LIST[@]}"; do
  echo "Copying bringup_ws: $ws"
  rm -rf /root/bringup_ws/src/$ws   # 💥 반드시 기존 경로 제거
  cp -r /workspace/bringup_ws/src/$ws /root/bringup_ws/src/$ws
done

# Build third-party library if it exists
if [ -d /opt/third_party/rbpodo/build ]; then
  echo "Building third_party: rbpodo"
  cd /opt/third_party/rbpodo/build
  make install
elif [ -d /opt/third_party/rbpodo ]; then
  echo "Building third_party: rbpodo"
  cd /opt/third_party/rbpodo
  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j$(nproc)
  make install
fi

cd /root/bringup_ws

if [ ! -f /root/bringup_ws/install/setup.bash ]; then
  echo "Building bringup_ws!!!!"
  colcon build --symlink-install
fi

echo 'source /root/bringup_ws/install/setup.bash' >> /root/.bashrc
source /root/bringup_ws/install/setup.bash

cd /root/ros2_ws
if [ ! -f /root/ros2_ws/install/setup.bash ]; then
  echo "Building ros2_ws!!!!"
  colcon build --symlink-install
fi
#colcon build --symlink-install
echo 'source /root/ros2_ws/install/setup.bash' >> /root/.bashrc

source /root/ros2_ws/install/setup.bash

echo "COMMON BUILD Done!"

ros2 launch mf_bringup bringup_manual_mode.launch.py
echo "RUN BRINGUP - ros2 launch mf_bringup bringup_manual_mode.launch.py"


exec "/bin/bash"
# while true; do sleep 3600; done
