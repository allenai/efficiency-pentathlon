import subprocess as sp
import sys
import os
from threading import Timer
import csv

def get_gpu_status():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --id=0 --query-gpu=timestamp,memory.used,power.draw --format=csv"
    try:
        gpu_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    # gpu_info_list = gpu_info[0].split(",")
    # gpu_info = f"Timestamp: {gpu_info_list[0]}; Memory: {gpu_info_list[1]}; Power: {gpu_info_list[2]}"
    gpu_info = gpu_info[0].split(",")
    return gpu_info


def write_gpu_status(writer, fout):
    gpu_info = get_gpu_status()
    print(gpu_info)
    writer.writerow(gpu_info)
    fout.flush()

if __name__ == "__main__":
    print(sys.argv)

    header = ["Timestamp", "Memory", "Power"]
    fout = open("log.csv", "w", encoding="UTF8", newline="")
    writer = csv.writer(fout)
    writer.writerow(header)
    timer = Timer(0.1, write_gpu_status, [writer, fout])
    timer.start()
    # write_gpu_status(writer, fout)
    # os.execvp(sys.argv[1], sys.argv[1:])
    os.system(" ".join(sys.argv[1:]))
    timer.cancel()
    fout.close()
