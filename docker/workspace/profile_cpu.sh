#!/bin/bash

while true; do docker stats $CONTAINER_ID --no-stream --format "{{.CPUPerc}} {{.MemPerc}}" | tee --append cpu.log; sleep 1; done