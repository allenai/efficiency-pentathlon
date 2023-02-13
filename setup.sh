#! /bin/bash

cd docker
docker build -t cpu_profiler .

cd ../rprof
pip install -e .

cd ../catwalk
pip install -e .
