#!/usr/bin/env python
import csv
import json
import multiprocessing
import os
import pathlib
import signal
import subprocess
import sys
import time
from carbon import get_realtime_carbon
from utils import LOG_DIR
import docker
import rprof

def profile_gpu():
    rprof.start_profiling(100, 99999)
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
        "python3 workspace/profile_cpu.py",
        name="test",
        privileged=True,
        tty=True,
        remove=True,
        volumes={
            f"{os.getcwd()}/workspace": {"bind": "/home/workspace", "mode": "rw"}
        },
        detach=True,
        stdout=True,
        stderr=True
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
    container.kill(signal.SIGINT)
    cpu_results = json.loads(container
                             .logs()
                             .strip()
                             .decode('UTF-8')
                             .replace("\'", "\""))
    p_gpu.wait()
    container.stop()

    gpu_energy, max_gpu_mem = 0, 0
    with open(f"{LOG_DIR}/gpu.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gpu_energy = gpu_energy + float(row["energy"])
            max_gpu_mem = max_gpu_mem + float(row["max_mem"])
    gpu_energy = gpu_energy / 3600.0
    cpu_energy = cpu_results["cpu_energy"] / 3600.0  # Wh
    mem_energy = cpu_results["dram_energy"] / 3600.0  # Wh
    # total_memory = subprocess.getoutput("cat /proc/meminfo | grep MemTotal")  # in KiB
    # total_memory = float(total_memory.split()[1]) / 2 ** 20
    time_elapsed = end_time - start_time
    total_energy = gpu_energy + cpu_energy + mem_energy
    carbon = get_realtime_carbon(total_energy)  # in g
    print(f"Time Elapsed: {time_elapsed:.2f} s")
    # print(f"Max DRAM Memory Usage: {max_mem_util * total_memory: .2f} GiB")
    print(f"Max GPU Memory Usage: {max_gpu_mem: .2f} GiB")
    print(f"GPU Energy: {gpu_energy:.2e} Wh")
    print(f"CPU Energy: {cpu_energy: .2e} Wh")
    print(f"Memory Energy: {mem_energy: .2e} Wh")
    print(f"Total Energy: {total_energy: .2e} Wh")
    print(f"CO2 emission: {carbon: .2e} grams.")
