#!/usr/bin/env python

import pylikwid
import time
import sys
import os

if __name__ == "__main__":
    pylikwid.inittopology()
    topodict = pylikwid.getcputopology()

    cpus = []
    for i in range(topodict["numSockets"]):
        cpus.append(i * topodict["numCoresPerSocket"])
    pinfo = pylikwid.getpowerinfo()
    cpu_domainid = pinfo["domains"]["PKG"]["ID"]
    dram_domainid = pinfo["domains"]["DRAM"]["ID"]
    pylikwid.init(cpus)
    cpu_starts, cpu_stops = [], []
    dram_starts, dram_stops = [], []
    for cpu in cpus:
        cpu_starts.append(pylikwid.startpower(cpu, cpu_domainid))
        dram_starts.append(pylikwid.startpower(cpu, dram_domainid))
    print(" ".join(sys.argv[1:]))
    os.system(" ".join(sys.argv[1:]))
    for cpu in cpus:
        cpu_stops.append(pylikwid.stoppower(cpu, cpu_domainid))
        dram_stops.append(pylikwid.stoppower(cpu, dram_domainid))
    for i in range(len(cpus)):
        print(f"CPU {i}: {pylikwid.getpower(cpu_starts[i], cpu_stops[i], cpu_domainid)}")
        print(f"DRAM {i}: {pylikwid.getpower(dram_starts[i], dram_stops[i], dram_domainid)}")
    
    