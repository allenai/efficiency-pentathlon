import subprocess
import signal
import os
import time
import sys

fout = open('log', 'w')
p = subprocess.Popen([f"{sys.executable}", "profile_cpu.py"], stdout=fout, stderr=fout, shell=False)
# Get the process id
pid = p.pid
time.sleep(10)
os.kill(pid, signal.SIGTERM)

if not p.poll():
    print("Process correctly halted")
fout.close()