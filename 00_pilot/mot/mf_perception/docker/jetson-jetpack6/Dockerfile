FROM ultralytics/ultralytics:latest-jetson-jetpack6

ENV DEBIAN_FRONTEND=noninteractive

## Update and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    cmake \
    wget \
    build-essential \
    libasio-dev \
    libtinyxml2-dev \
    software-properties-common && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

## Install RealSense SDK
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list
RUN apt-get update  -y && \
    apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg

## Install ROS2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && \
    apt-get install -y --fix-missing \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-flake8-docstrings \
    python3-pytest-cov \
    ros-dev-tools \
    dbus-x11

## Install ROS2 packages
RUN apt-get install -y \
    libboost-thread-dev \
    libpoco-dev \
    ros-humble-serial-driver \
    ros-humble-asio-cmake-module \
    ros-humble-usb-cam \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-vision-msgs* \
    ros-humble-tf* \
    ros-humble-control-msgs \
    ros-humble-realtime-tools \
    ros-humble-xacro \
    ros-humble-joint-state-publisher-gui \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gazebo-msgs \
    ros-humble-moveit-msgs \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-ros-move-group \
    ros-humble-srdfdom \
    ros-humble-ament-cmake \
    ros-humble-joint-state-publisher \
    ros-humble-moveit \
    ros-humble-pluginlib \
    ros-humble-robot-state-publisher \
    ros-humble-ros2-controllers \
    ros-humble-ros2-control \
    ros-humble-rviz2 \
    ros-humble-urdf-launch \
    ros-humble-xacro \
    ros-humble-geometric-shapes \
    ros-humble-srdfdom

    
SHELL ["/bin/bash", "-c"]
RUN mkdir /root/ros2_ws
WORKDIR /root/ros2_ws
RUN cd /root/ros2_ws
RUN rosdep init && rosdep update
RUN rosdep install -r --from-paths . --ignore-src --rosdistro humble -y

RUN source /opt/ros/humble/setup.bash && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools==58.2.0

RUN python3 -m pip install \
    lapx \
    squaternion \
    cython_bbox \
    pyserial \
    numpy==1.25 \
    numpy-quaternion \
    pykdtree \
    pytictoc \
    matplotlib==3.5.1


# COPY ./.bashrc /root
RUN echo "export PYTHONPATH='$PYTHONPATH:/opt/:/workspace/'" >> ~/.bashrc && \
echo "alias ccd='colcon_cd '" >> ~/.bashrc && \
    echo "alias cb='cd ~/ros2_ws && colcon build --symlink-install'" >> ~/.bashrc && \
    echo "alias cbs='colcon build --symlink-install'" >> ~/.bashrc && \
    echo "alias cbp='colcon build --symlink-install --packages-select'" >> ~/.bashrc && \
    echo "alias cbu='colcon build --symlink-install --packages-up-to'" >> ~/.bashrc && \
    echo "alias ct='colcon test'" >> ~/.bashrc && \
    echo "alias ctp='colcon test --packages-select'" >> ~/.bashrc && \
    echo "alias ctr='colcon test-result '" >> ~/.bashrc && \
    echo "alias cw='cd /root/ros2_ws'" >> ~/.bashrc && \
    echo "alias cs='cd /root/ros2_ws/src'" >> ~/.bashrc && \
    echo "alias eb='gedit ~/.bashrc'" >> ~/.bashrc && \
    echo "alias sb='source ~/.bashrc'" >> ~/.bashrc && \
    echo "alias python='python3'" >> ~/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=7" >> ~/.bashrc && \
    echo "export ROS_NAMESPACE=robot1" >> ~/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc


## Piper installation
# RUN apt-get install -y \
#             iproute2 \
#             can-utils
# RUN echo "ip link set can0 up type can bitrate 1000000" >> ~/.bashrc

# RUN apt-get update && apt-get install -y \
#     iproute2 \
#     can-utils \
#     net-tools \
#     build-essential \
#     cmake \
#     git


# ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && bash"]
