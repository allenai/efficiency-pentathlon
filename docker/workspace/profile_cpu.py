#!/usr/bin/env python

import pylikwid
import time
import sys
import os
import signal


def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)
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
    start = time.time()
    try:
        while True:
            time.sleep(0.1)
    finally:
        for cpu in cpus:
            cpu_stops.append(pylikwid.stoppower(cpu, cpu_domainid))
            dram_stops.append(pylikwid.stoppower(cpu, dram_domainid))
        with open("cpu_log", "w") as fout:
            fout.write(f"Time elapsed: {time.time() - start}s\n")
            for i in range(len(cpus)):
                fout.write(f"CPU {i}: {pylikwid.getpower(cpu_starts[i], cpu_stops[i], cpu_domainid)}\n")
                fout.write(f"DRAM {i}: {pylikwid.getpower(dram_starts[i], dram_stops[i], dram_domainid)}\n")
        sys.exit()

    
    