#!/usr/bin/env python
import time
import sys
import os
import subprocess
import signal
import docker
import time


if __name__ == "__main__":
    client = docker.from_env()
    devnull = open('/dev/null', 'w')
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
    p = subprocess.Popen([f"{sys.executable}", "workspace/profile_gpu.py"], stdout=devnull, shell=False)

    print(" ".join(sys.argv[1:]))
    os.system(" ".join(sys.argv[1:]))
    #####
    os.kill(p.pid, signal.SIGTERM)
    if not p.poll():
        print("Process correctly halted")
    container.kill("SIGTERM")
    for line in output:
        print(line)
    