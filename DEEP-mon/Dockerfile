FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
LABEL maintainer="Rolando Brondolin"

RUN apt-get clean && apt-get update \
  && apt-get install -y python3 python3-pip locales locales-all libelf1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && pip3 install numpy pyyaml docker


#Needed by Curse to print unicode characters to the terminal
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

#install bcc and ddsketch

RUN buildDeps='python python3-pip python3-distutils python3-apt wget curl apt-transport-https git bison build-essential cmake flex libedit-dev libllvm12 llvm-12-dev libclang-12-dev zlib1g-dev libelf-dev' \
  && apt-get update && apt-get install -y $buildDeps

  # && git checkout 043478d664003738a11043b81005f623e4b4910a\
RUN git clone https://github.com/iovisor/bcc.git \
  && cd bcc \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j \
  && make install \
  && cmake -DPYTHON_CMD=python3 .. \
  && cd src/python/ \
  && make \
  && make install \
  && cd /
  # && rm -r bcc

RUN git clone --branch v1.0 https://github.com/DataDog/sketches-py.git \
  && cd sketches-py \
  && python3 setup.py install \
  && cd / \
  && rm -r sketches-py \
  && apt-get purge -y --auto-remove $buildDeps \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /home

RUN mkdir /home/deep_mon
ADD bpf /home/deep_mon/bpf
ADD userspace /home/deep_mon/userspace
ADD deep_mon.py /home/deep_mon/
ADD setup.py /home

RUN apt-get update && apt-get install -y python3-distutils
RUN apt-get install -y vim
#Install plugin dependencies
RUN pip3 install .  
# && rm -rf /home/deep_mon && rm setup.py

# "-u" forces the binary I/O layers of stdout and stderr to be unbuffered and
# is needed to avoid truncated output in Docker
ENV PYTHONUNBUFFERED="on"
CMD ["deep-mon"]
