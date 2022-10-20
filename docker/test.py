import subprocess
import signal
import os
import time
import sys

devnull = open('/dev/null', 'w')
p = subprocess.Popen([f"{sys.executable}", "profile_gpu.py"], stdout=devnull, shell=False)
# Get the process id
pid = p.pid

time.sleep(5)
os.kill(pid, signal.SIGTERM)

if not p.poll():
    print("Process correctly halted")