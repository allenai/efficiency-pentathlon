#!/usr/bin/env python

import pylikwid
import time
import sys
import os
import signal


def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)


if __name__ == "__main__":
    init_start = time.time()
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
    init_time = start - init_start
    print(f"Init time: {init_time:.3f}s")
    try:
        while True:
            time.sleep(0.1)
    finally:
        time.sleep(init_time)
        for cpu in cpus:
            cpu_stops.append(pylikwid.stoppower(cpu, cpu_domainid))
            dram_stops.append(pylikwid.stoppower(cpu, dram_domainid))
        time_elapsed = time.time() - start
        with open(f"{os.getcwd()}/workspace/log/cpu.csv", "w") as fout:
            print(f"Time elapsed: {time_elapsed:.3f}s\n")
            fout.write("id,time_elapsed,cpu_energy,dram_energy\n")
            for i in range(len(cpus)):
                cpu_energy = pylikwid.getpower(cpu_starts[i], cpu_stops[i], cpu_domainid)
                dram_energy = pylikwid.getpower(dram_starts[i], dram_stops[i], dram_domainid)
                print(f"CPU {i}: {cpu_energy:.3f}\n")
                print(f"DRAM {i}: {dram_energy:.3f}\n")
                fout.write(f"{i},{time_elapsed:.3f},{cpu_energy:.3f},{dram_energy:.3f}\n")
        sys.exit()

    
    