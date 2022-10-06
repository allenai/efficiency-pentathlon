#!/bin/bash

set -eo pipefail
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda env create --name efficiency \
   --file /workspace/environment.yml
conda activate efficiency

# Install PowerTOP
# git clone https://github.com/fenrus75/powertop.git
# cd powertop
# ./autogen.sh
# ./configure
# make && make install

# Install Pyroscope
# exec the final command:
# exec python run.py