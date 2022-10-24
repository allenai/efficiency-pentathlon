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


if __name__ == "__main__":
    num_cpus = multiprocessing.cpu_count()
    client = docker.from_env()
    devnull = open("/dev/null", "w")
    container = client.containers.run(
        "test:latest",
        "python workspace/profile_cpu.py",
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
    p_docker = subprocess.Popen([f"sh", "docker_monitor.sh"], stdout=devnull, shell=False)

    print(" ".join(sys.argv[1:]))
    start_time = time.time()
    os.system(" ".join(sys.argv[1:]))
    end_time = time.time()
    os.kill(p_gpu.pid, signal.SIGTERM)
    if not p_gpu.poll():
        print("GPU process correctly halted")
    os.kill(p_docker.pid, signal.SIGTERM)
    if not p_docker.poll():
        print("Docker process correctly halted")
    container.kill("SIGTERM")
    cpu_energy, mem_energy = 0, 0
    for line in output:
        ts = line.split()
        if len(ts) > 5:
            cpu_energy = cpu_energy + float(ts[5]) + float(ts[11])
            mem_energy = mem_energy + float(ts[8]) + float(ts[14])

    with open("workspace/log/gpu.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        gpu_energy, max_mem = 0, 0
        for row in reader:
            gpu_energy = gpu_energy + float(row["energy"])
            max_mem = max_mem + float(row["max_mem"])
    # with open("workspace/log/cpu.csv") as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     cpu_energy, mem_energy = 0, 0
    #     for row in reader:
    #         cpu_energy = cpu_energy + float(row["cpu_energy"])
    #         mem_energy = mem_energy + float(row["dram_energy"])
    with open("workspace/log/docker.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        cpu_util, mem_util, max_mem = 0, 0, 0
        num_rows = 1e-6
        for row in reader:
            num_rows += 1
            cpu_util = cpu_util + float(row["cpu_util"].strip("%")) / 100
            mem_util = mem_util + float(row["mem_util"].strip("%")) / 100
            # max_mem = max(max_mem, float(row["mem"]))
        cpu_util = cpu_util / (num_rows) / num_cpus
        mem_util = mem_util / (num_rows)
    cpu_energy = cpu_energy * cpu_util
    mem_energy = mem_energy * mem_util
    print(f"Time Elapsed: {end_time - start_time:.3f}s", end="; ") 
    print(f"GPU Energy: {gpu_energy:.3f}W.s", end="; ")
    print(f"CPU Energy: {cpu_energy: .3f}W.s", end="; ")
    print(f"Memory Energy: {mem_energy: .3f}W.s", end="; ")
    print(f"Total Energy: {gpu_energy + cpu_energy + mem_energy: .3f}W.s")
    