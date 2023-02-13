#!/usr/bin/env python

import pylikwid
import time
import sys
import signal


TIME_INTERVAL = 0.1

if __name__ == "__main__":
    init_start = time.time()
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

    start = time.time()
    def _signal_handler(sig, _frame):
        for cpu in cpus:
            cpu_stops.append(pylikwid.stoppower(cpu, cpu_domainid))
            dram_stops.append(pylikwid.stoppower(cpu, dram_domainid))
        time_elapsed = time.time() - start
        results = {
            "time": time_elapsed,
            "cpu_energy": 0.0,
            "dram_energy": 0.0
        }
        for i in range(len(cpus)):
            cpu_energy = pylikwid.getpower(cpu_starts[i], cpu_stops[i], cpu_domainid)
            dram_energy = pylikwid.getpower(dram_starts[i], dram_stops[i], dram_domainid)
            results[f"cpu_energy"] += cpu_energy  # Ws
            results[f"dram_energy"] += dram_energy  # Ws
        print(results)
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    for cpu in cpus:
        cpu_starts.append(pylikwid.startpower(cpu, cpu_domainid))
        dram_starts.append(pylikwid.startpower(cpu, dram_domainid))
    signal.pause()
