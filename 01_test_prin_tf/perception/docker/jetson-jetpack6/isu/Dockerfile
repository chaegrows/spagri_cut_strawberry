FROM ultralytics/ultralytics:latest-jetson-jetpack6

RUN apt-get update

RUN mkdir -p /workspace

WORKDIR /workspace

COPY ./install_glim_in_docker.sh /workspace
COPY docker/entrypoint.sh /entrypoint.sh

RUN apt-get install 
RUN /bin/bash -c '/workspace/install_glim_in_docker.sh'
