FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV ENV_NAME=efficiency
# Codebase
# COPY distill /

ARG DEBIAN_FRONTEND=noninteractive
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt update --fix-missing -y && \
    apt upgrade -y && \
    apt install -y \
    ca-certificates \
    gcc \
    g++ \
    git \
    sudo \
    curl \
    libx11-6 \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Create a working directory
RUN mkdir /app
WORKDIR /app

COPY environment.yml /app/environment.yml
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# SHELL ["/bin/bash", "--login"]
# RUN eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
RUN conda env create --name ${ENV_NAME} --file /app/environment.yml

# Install likwid to monitor CPU and DRAM energy
# https://github.com/RRZE-HPC/likwid.git
RUN git clone https://github.com/RRZE-HPC/likwid.git \
    && cd likwid \
    && make && make install \
    && cd .. \ 
    && git clone https://github.com/RRZE-HPC/pylikwid.git \
    && cd pylikwid \
    && python setup.py build_ext -I /app/likwid/src/includes/ -L /app/likwid/ -R /app/likwid/ \
    && pip install -e . \
    && cd .. 

# Install rprof to monitor GPU energy
# COPY rprof ./rprof
# RUN cd rprof \
#     && pip install -e . \
#    && rm -rf rprof

# COPY profile_cpu.sh ./
# COPY hello_world.py ./
# COPY entrypoint.py ./
# COPY profile_gpu.py ./

# RUN conda activate ${ENV_NAME}
# ENTRYPOINT ["./entrypoint.sh"]