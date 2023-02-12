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
    time_interval = 0.1
    wrapup_time = 10
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
        "python3 workspace/cpu_power.py",
        name="test",
        privileged=True,
        tty=True,
        remove=True,
        volumes={
            f"{os.getcwd()}/workspace": {"bind": "/home/workspace", "mode": "rw"}
        },
        detach=True
    )
    output = container.attach(stdout=True, stream=True, logs=True, stderr=True)
    p_gpu = subprocess.Popen([f"{sys.executable}", "workspace/profile_gpu.py"], stdout=devnull, shell=False)

    def wrapup():
        print(f"Sleep for {wrapup_time} seconds...")
        time.sleep(wrapup_time)
        os.chdir(monitor_dir)
        os.kill(p_gpu.pid, signal.SIGTERM)
        if not p_gpu.poll():
            print("GPU monitor correctly halted")
        container.kill("SIGTERM")
        p_gpu.wait()

        timestamp = 0
        cpu_power, gpu_power, total_power = {}, {}, {}
        with open(f"{LOG_DIR}/gpu_power.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gpu_power[timestamp] = float(row['gpu'])
                timestamp += 1
        max_time = timestamp
        timestamp = 0
        with open("workspace/log/cpu_power.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cpu_power[timestamp] = float(row['cpu']) + float(row['dram'])
                timestamp += 1
        max_time = min(max_time, timestamp)
        timestamp = 0
        with open(f"calibration_data/{sys.argv[-1]}", "w") as fout:
            fout.write("time,power\n")
            while timestamp < max_time:
                assert timestamp in cpu_power and timestamp in gpu_power
                total_power[timestamp] = cpu_power[timestamp] + gpu_power[timestamp]
                print(f"{timestamp * time_interval : .1f}, {total_power[timestamp]: .2f}")
                fout.write(f"{timestamp * time_interval},{total_power[timestamp]}\n")
                timestamp += 1

    def sigterm_handler(_signo, _stack_frame):
        wrapup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    print("Executing:", " ".join(sys.argv[1:]))
    os.chdir(cur_dir)
    start_time = time.time()
    time.sleep(5)
    os.system(" ".join(sys.argv[1:]))
    end_time = time.time()
    wrapup()
