#! /bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -y -n efficiency-benchmark python=3.9
conda activate efficiency-benchmark

conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# cd docker
# docker build -t cpu_profiler .

cd rprof
pip install -e .

cd ../catwalk
pip install -e .
