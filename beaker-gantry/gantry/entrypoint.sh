#!/bin/bash

set -eo pipefail

cd ../../docker
docker build -t test .
docker run --gpus device=0 -it --privileged test /bin/bash
# python rprof/profile_gpu.py & 
# RPROF_PID=$!

# sh profile_cpu.sh &
# CPU_PID=$!
# while true; do docker stats $CONTAINER_ID --no-stream --format "{{.CPUPerc}} {{.MemPerc}}" | tee --append stats.txt; sleep 1; done

# python execute.py "$@" 2>&1 | tee "${{ RESULTS_DIR }}/.gantry/out.log"
