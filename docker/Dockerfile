FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV ENV_NAME=efficiency

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

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
    wget \
    curl \
    apt-transport-https \
    bison \
    build-essential \
    cmake \
    flex \
    libedit-dev \
    libllvm12 \
    llvm-12-dev \
    libclang-12-dev \
    zlib1g-dev \
    libelf-dev \
    locales \
    locales-all \
    libelf1 \
    python3 \
    python3-pip \
    && apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh

WORKDIR /home
# ADD environment.yml /home/environment.yml
# ADD entrypoint.sh /home/entrypoint.sh
# RUN chmod +x entrypoint.sh
# RUN conda env create --name ${ENV_NAME} --file /home/environment.yml

# RUN git clone https://github.com/iovisor/bcc.git \
#   && cd bcc \
#   && mkdir build \
#   && cd build \
#   && cmake .. \
#   && make -j \
#   && make install \
#   && cmake -DPYTHON_CMD=python3 .. \
#   && cd src/python/ \
#   && make -j \
#   && make install \
#   && cd /home \
#   && rm -r bcc

# RUN git clone --branch v1.0 https://github.com/DataDog/sketches-py.git \
#   && cd sketches-py \
#   && python3 setup.py install \
#   && cd /home \
#   && rm -r sketches-py \
#   && rm -rf /var/lib/apt/lists/*

# Install likwid to monitor CPU and DRAM energy
# https://github.com/RRZE-HPC/likwid.git
RUN git clone https://github.com/RRZE-HPC/likwid.git \
    && cd likwid \
    && make -j && make install \
    && cd .. \ 
    && git clone https://github.com/RRZE-HPC/pylikwid.git \
    && cd pylikwid \
    && python3 setup.py build_ext -I /home/likwid/src/includes/ -L /home/likwid/ -R /home/likwid/ \
    && pip3 install -e . \
    && cd /home


# RUN mkdir /home/deep_mon
# ADD bpf /home/deep_mon/bpf
# ADD userspace /home/deep_mon/userspace
# ADD deep_mon.py /home/deep_mon/
# ADD setup.py /home
# ADD config.yaml /home
# ADD cpu.py /home

# RUN pip3 install --editable . 
# RUN pip3 install --upgrade numpy pyyaml docker
ENV PYTHONUNBUFFERED="on"
# RUN conda activate ${ENV_NAME}
ADD profile_cpu.py /home
# ENTRYPOINT ["python3", "profile_cpu.py"]
