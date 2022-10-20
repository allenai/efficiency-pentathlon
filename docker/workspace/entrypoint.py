#!/usr/bin/env python

import sys
import os
import subprocess
import signal

if __name__ == "__main__":
    devnull = open('/dev/null', 'w')
    p = subprocess.Popen([f"{sys.executable}", "profile_gpu.py"], stdout=devnull, shell=False)
    #####
    os.kill(p.pid, signal.SIGTERM)
    if not p.poll():
        print("Process correctly halted")
    
    
    