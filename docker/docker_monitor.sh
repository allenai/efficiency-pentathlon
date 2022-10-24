#!/bin/bash
CONTAINER_ID=$(basename $(cat /proc/1/cpuset))
OUTPUT_PATH="workspace/log/docker.csv"
rm -f workspace/log/docker.csv
echo "cpu_util,mem_util,mem" > $OUTPUT_PATH
while true; do 
    stdbuf --output=0 docker stats $CONTAINER_ID --no-stream --format "{{.CPUPerc}},{{.MemPerc}},{{.MemUsage}}" | tee -a $OUTPUT_PATH; 
    sleep 1; 
done