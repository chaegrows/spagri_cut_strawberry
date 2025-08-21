# [Base image with CUDA and CUDNN]
FROM ultralytics/ultralytics:8.2.41
# Add GPG key using new method
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /etc/apt/keyrings/coral-edgetpu-archive-keyring.gpg

# Update source list to use new keyring
RUN echo "deb [signed-by=/etc/apt/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list


# [Docker image build setup]
ARG IMAGE_NAME='ubuntu2204_cuda12_ros2humble'
ARG ROS_DISTRO='humble'

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS yes

# Conda directory
RUN rm -rf /opt/conda

# [Install dependencies via apt]
RUN set -x && \
apt-get update -y -qq && \
apt-get install -y -qq \
    libyaml-cpp-dev curl software-properties-common \
    gnupg2 \
    lsb-release \
    cmake \
    wget \
    build-essential \
    libasio-dev \
    libtinyxml2-dev \
    locales \
    gedit \
    bash-completion \
    gnupg lsb-release \
    gedit git vim v4l-utils exfat-* \
    openssh-server \
    terminator dbus-x11 \
    python3-pip \
    iproute2 \
    can-utils \
    net-tools && \
: "remove cache" && \
apt-get autoremove -y -qq && \
rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8


# Install RealSense SDK
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list
RUN apt-get update  -y && \
    apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg



# [Install Cmake]
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt update && \
    apt install -y -qq cmake


# [Install ROS2 Humble]
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update && \
    apt upgrade -y && \
    apt install -q -y \
    build-essential \
    libasio-dev \
    libtinyxml2-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-flake8-docstrings \
    python3-pytest-cov \
    ros-dev-tools \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-*-msgs \
    ros-${ROS_DISTRO}-control* \
    ros-${ROS_DISTRO}-usb-cam \
    ros-${ROS_DISTRO}-realsense2-camera \
    ros-${ROS_DISTRO}-realsense2-description \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-ur* \
    ros-${ROS_DISTRO}-librealsense2* \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-vision-msgs*


RUN apt update && \
    apt install -q -y \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-tf2-tools \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-nav* \
    ros-${ROS_DISTRO}-slam-toolbox* \
    ros-${ROS_DISTRO}-tf* \
    ros-${ROS_DISTRO}-serial* \
    ros-${ROS_DISTRO}-asio-cmake-module \
    ros-${ROS_DISTRO}-moveit*




RUN python3 -m pip install -U \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-import-order \
    flake8-quotes \
    "pytest>=5.3" \
    pytest-repeat \
    pytest-rerunfailures
RUN apt update && \
    apt install -q -y \
    libfmt-dev librange-v3-dev
RUN pip install -U pip && \
    pip install -U jinja2 typeguard seaborn

RUN mkdir -p ~/ros2_ws/src && \
    cd ~/ros2_ws && \
    rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers" && \
    colcon build --symlink-install
RUN echo 'source /root/ros2_ws/install/setup.bash' >> ~/.bashrc
RUN apt install -y net-tools
RUN apt install -y ros-humble-py-trees*


COPY docker/${IMAGE_NAME}/dependencies/git-completion.bash /root
COPY docker/${IMAGE_NAME}/dependencies/git-prompt.sh /root
COPY docker/${IMAGE_NAME}/dependencies/bash.bashrc /etc
COPY docker/${IMAGE_NAME}/dependencies/.bashrc /root
COPY docker/${IMAGE_NAME}/dependencies/colcon_cd.sh /root
COPY docker/${IMAGE_NAME}/dependencies/colcon-argcomplete.bash /root
COPY docker/${IMAGE_NAME}/dependencies/vcs.bash /root
COPY docker/${IMAGE_NAME}/dependencies/my_robot_calibration.yaml /root
RUN /bin/bash -c "source /etc/bash.bashrc" && \
    rm /etc/apt/apt.conf.d/docker-clean && \
    apt update -y

COPY docker/${IMAGE_NAME}/dependencies/requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed -r /tmp/requirements.txt
RUN pip install --no-cache-dir psutil==6.0.0 dill


RUN apt-get update
RUN apt-get install -y libpoco-dev libyaml-cpp-dev wget


WORKDIR /root/ros2_ws/src

ENTRYPOINT ["/bin/bash"]
