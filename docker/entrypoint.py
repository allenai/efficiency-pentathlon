#!/usr/bin/env python

import pylikwid
import time
import sys
import os
import subprocess
import signal

if __name__ == "__main__":
    devnull = open('/dev/null', 'w')
    pylikwid.inittopology()
    topodict = pylikwid.getcputopology()

    cpus = []
    for i in range(topodict["numSockets"]):
        cpus.append(i * topodict["numCoresPerSocket"])
    pinfo = pylikwid.getpowerinfo()
    cpu_domainid = pinfo["domains"]["PKG"]["ID"]
    dram_domainid = pinfo["domains"]["DRAM"]["ID"]
    pylikwid.init(cpus)
    e_starts, e_stops = [], []
    for cpu in cpus:
        e_starts.append(pylikwid.startpower(cpu, domainid))
    
    p = subprocess.Popen([f"{sys.executable}", "profile_gpu.py"], stdout=devnull, shell=False)
    print(" ".join(sys.argv[1:]))
    os.system(" ".join(sys.argv[1:]))
    os.kill(p.pid, signal.SIGTERM)
    if not p.poll():
        print("Process correctly halted")
    for cpu in cpus:
        e_stops.append(pylikwid.stoppower(cpu, domainid))
    for i in range(len(cpus)):
        print(f"CPU {i}: {pylikwid.getpower(e_starts[i], e_stops[i], domainid)}")