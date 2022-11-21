"""
    DEEP-mon
    Copyright (C) 2020  Brondolin Rolando

    This file is part of DEEP-mon

    DEEP-mon is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DEEP-mon is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import multiprocessing
import os
import ctypes as ct

class BpfProcTopology(ct.Structure):
    _fields_ = [("ht_id", ct.c_ulonglong),
                ("sibling_id", ct.c_ulonglong),
                ("core_id", ct.c_ulonglong),
                ("processor_id", ct.c_ulonglong),
                ("cycles_core", ct.c_ulonglong),
                ("cycles_core_delta_sibling", ct.c_ulonglong),
                ("cycles_thread", ct.c_ulonglong),
                ("cache_misses", ct.c_ulonglong),
                ("cache_refs", ct.c_ulonglong),
                ("instr_thread", ct.c_ulonglong),
                ("ts", ct.c_ulonglong),
                ("running_pid", ct.c_int)]

class ProcTopology:
    processors_path = '/proc/cpuinfo'

    def __init__(self):
        # parse /proc/cpuinfo to obtain processor topology
        ht_id = 0
        sibling_id = 0
        core_id = 0
        processor_id = 0

        #core elem is organized as ht_id, sibling_id, core_id, processor_id
        self.coresDict = {}
        self.socket_set = set()

        with open(ProcTopology.processors_path) as f:
            for line in f:
                sp = line.split(" ")
                if "processor\t" in sp[0]:
                    ht_id = int(sp[1])
                if "physical" in sp[0] and "id\t" in sp[1]:
                    processor_id = int(sp[2])
                    self.socket_set.add(processor_id)
                if "core" in sp[0] and "id\t\t" in sp[1]:
                    core_id = int(sp[2])
                    found = False
                    for key, value in self.coresDict.items():
                        if value[2] == core_id and value[3] == processor_id:
                            found = True
                            value[1] = ht_id
                            self.coresDict[ht_id] = [ht_id, value[0], core_id, \
                                processor_id]
                            break
                    if not found:
                        self.coresDict[ht_id] = [ht_id, -1, core_id, processor_id]

    def print_topology(self):
        for key, value in self.coresDict.items():
            print(value)

    def get_topology(self):
        return self.coresDict

    def get_sockets(self):
        return self.socket_set

    def get_hyperthread_count(self):
        return len(self.coresDict)

    def get_new_bpf_topology(self):
        bpf_dict = {}
        for key,value in self.coresDict.items():
            core = BpfProcTopology(ct.c_ulonglong(value[0]), \
                                ct.c_ulonglong(value[1]), \
                                ct.c_ulonglong(value[2]), \
                                ct.c_ulonglong(value[3]), \
                                ct.c_ulonglong(0), \
                                ct.c_ulonglong(0), \
                                ct.c_ulonglong(0), \
                                ct.c_ulonglong(0), \
                                ct.c_ulonglong(0), \
                                ct.c_ulonglong(0), \
                                ct.c_ulonglong(0), \
                                ct.c_int(0))
            bpf_dict[key] = core
        return bpf_dict
