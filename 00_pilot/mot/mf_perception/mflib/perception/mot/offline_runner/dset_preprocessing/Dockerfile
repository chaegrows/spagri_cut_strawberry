FROM koide3/glim_ros2:humble

# Install dependencies
# suppress shell warnings
RUN apt-get update \
  && apt-get install --no-install-recommends -y \
    libfmt-dev libspdlog-dev libopencv-dev 
  # &&   apt-get clean \
  # && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
    python3-pip 

RUN pip3 install --no-cache-dir \
  numpy \
  opencv-python \
  scipy \
  numpy-quaternion

WORKDIR /root/dset_preprocessing

CMD ["/bin/bash"]