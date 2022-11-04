#!/usr/bin/env python
import time
import sys
import os
import subprocess
import signal
import docker
import time
import csv
import multiprocessing
from utils import get_num_instances
from utils import LOG_DIR
from carbon import get_realtime_carbon
import pathlib


if __name__ == "__main__":
    cur_dir = os.getcwd()
    monitor_dir = pathlib.Path(__file__).parent.resolve()
    os.chdir(monitor_dir)
    print(os.getcwd())
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    num_cpus = multiprocessing.cpu_count()
    client = docker.from_env()
    devnull = open("/dev/null", "w")
    container = client.containers.run(
        "test:latest",
        "python workspace/cpu_power.py",
        name="test",
        privileged=True,
        tty=True,
        remove=True,
        volumes={
            f"{os.getcwd()}/workspace": {"bind": "/app/workspace", "mode": "rw"}
        },
        detach=True
    )
    output = container.attach(stdout=True, stream=True, logs=True, stderr=True)
    p_gpu = subprocess.Popen([f"{sys.executable}", "workspace/profile_gpu.py"], stdout=devnull, shell=False)

    print("Executing:", " ".join(sys.argv[1:]))
    os.chdir(cur_dir)
    start_time = time.time()
    os.system(" ".join(sys.argv[1:]))
    end_time = time.time()
    os.chdir(monitor_dir)
    os.kill(p_gpu.pid, signal.SIGTERM)
    if not p_gpu.poll():
        print("GPU monitor correctly halted")
    container.kill("SIGTERM")
    p_gpu.wait()

    with open(f"{LOG_DIR}/gpu_power.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(float(row["gpu"]))
    with open("workspace/log/cpu_power.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(float(row["cpu"]) + float(row["dram"]))
    dsadas
    gpu_energy = gpu_energy / 3600.0
    cpu_energy = cpu_energy / 3600.0
    mem_energy = mem_energy / 3600.0
    time_elapsed = end_time - start_time
    total_energy = gpu_energy + cpu_energy + mem_energy
    carbon = get_realtime_carbon(total_energy)  # in grams

    print(f"Time Elapsed: {time_elapsed:.2f} s") 
    print(f"GPU Energy: {gpu_energy:.2e} Wh", end="; ")
    print(f"CPU Energy: {cpu_energy: .2e} Wh", end="; ")
    print(f"Memory Energy: {mem_energy: .2e} Wh", end="; ")
    print(f"Total Energy: {total_energy: .2e} Wh", end="; ")
    print(f"CO2 emission: {carbon: .2e} grams.")
    