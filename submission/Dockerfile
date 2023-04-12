ARG BASE_IMAGE=ubuntu:20.04

FROM ${BASE_IMAGE}

# TODO: might not need this line anymore
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ARG DEBIAN_FRONTEND="noninteractive"

RUN  apt-get update && \
     apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        libxml2-dev \
        jq \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace/
ADD . /workspace/
RUN pip install -r requirements.txt

# ENTRYPOINT ["/opt/conda/bin/conda"]
