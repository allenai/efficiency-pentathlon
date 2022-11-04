#!/usr/bin/env python

import pylikwid
import time
import sys
import os
import signal


def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)


if __name__ == "__main__":
    time_interval = 0.1
    init_start = time.time()
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    pylikwid.inittopology()
    topodict = pylikwid.getcputopology()

    cpus = []
    for i in range(topodict["numSockets"]):
        cpus.append(i * topodict["numCoresPerSocket"])
    pinfo = pylikwid.getpowerinfo()
    cpu_domainid = pinfo["domains"]["PKG"]["ID"]
    dram_domainid = pinfo["domains"]["DRAM"]["ID"]
    pylikwid.init(cpus)
    cpu_starts, cpu_stops = {}, {}
    dram_starts, dram_stops = {}, {}
    start = time.time()
    init_time = start - init_start

    fout = open(f"{os.getcwd()}/workspace/log/cpu_power.csv", "w")
    fout.write("time,cpu,dram\n")
    try: 
        while True:
            for j in range(len(cpus)):
                cpu = cpus[j]
                cpu_starts[j] = pylikwid.startpower(cpu, cpu_domainid)
                dram_starts[j] = pylikwid.startpower(cpu, dram_domainid)
            time.sleep(time_interval)
            current_time = time.time() - start
            for j in range(len(cpus)):
                cpu = cpus[j]
                cpu_stops[j] = pylikwid.stoppower(cpu, cpu_domainid)
                dram_stops[j] = pylikwid.stoppower(cpu, dram_domainid)
            
            for j in range(len(cpus)):
                cpu_energy = pylikwid.getpower(
                    cpu_starts[j], cpu_stops[j], cpu_domainid)
                dram_energy = pylikwid.getpower(
                    dram_starts[j], dram_stops[j], dram_domainid)
                cpu_power = cpu_energy / time_interval
                dram_power = dram_energy / time_interval
                print(f"Time: {current_time: .2f}; CPU: {cpu_power: .2f}W; DRAM: {dram_power: .2f}W")
                fout.write(f"{current_time},{cpu_power},{dram_power}\n")
    finally:
        fout.close()
    