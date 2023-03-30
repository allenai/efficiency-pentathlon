FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
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
    vim \
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

WORKDIR /home
# ADD requirements.txt requirements.txt
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install transformers
RUN pip3 install sentencepiece

ENV PYTHONUNBUFFERED="on"
ADD entrypoint.py /home
ENTRYPOINT ["python3", "entrypoint.py"]
