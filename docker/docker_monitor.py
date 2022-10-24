import docker
import subprocess

container_id = subprocess.getoutput("echo $(basename $(cat /proc/1/cpuset))")
total_memory = subprocess.getoutput("cat /proc/meminfo | grep MemTotal")  # in kB
total_memory = float(total_memory.split()[1])
client = docker.from_env()
container_cpu = 0
system_cpu = 0

container = client.containers.get(container_id)

#This function is blocking; the loop will proceed when there's a new update to iterate
for stats in container.stats(decode=True):
    print(stats)
    # print(stats['cpu_stats']['cpu_usage'])
    print(stats['cpu_stats']['system_cpu_usage'])
    print(stats['precpu_stats'])
    dsa
    print(stats['precpu_stats']['cpu_usage']['system_cpu_usage'])
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
    system_delta = stats['cpu_stats']['cpu_usage']['system_cpu_usage'] - stats['precpu_stats']['cpu_usage']['system_cpu_usage']
    len_cpu = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])
    cpu_util = (cpu_delta / system_delta) * len_cpu * 100
    mem_used = (stats["memory_stats"]["usage"] 
                    - stats["memory_stats"]["stats"]["cache"] 
                    + stats["memory_stats"]["stats"]["active_file"])
    mem_util = mem_used / 1024 / total_memory * 100
    print(f"{cpu_util:.3f}, {mem_util:.3f}")








    # #Save the values from the last sample
    # prev_container_cpu = container_cpu
    # prev_system_cpu = system_cpu

    # #Get the container's usage, the total system capacity, and the number of _cpus
    # #The math returns a Linux-style %util, where 100.0 = 1 _cpu core fully used
    # container_cpu = stats.get('cpu_stats',{}).get('cpu_usage',{}).get('total_usage')
    # system_cpu    = stats.get('cpu_stats',{}).get('system_cpu_usage')
    # num_cpu   = len(stats.get('cpu_stats',{}).get('cpu_usage',{}).get('percpu_usage',0))

    # # Skip the first sample (result will be wrong because the saved values are 0)
    # if prev_container_cpu and prev_system_cpu:
    #     cpu_util = (container_cpu - prev_container_cpu) / (system_cpu - prev_system_cpu)
    #     cpu_util = cpu_util * num_cpu * 100
    #     mem_used = (stats["memory_stats"]["usage"] 
    #                 - stats["memory_stats"]["stats"]["cache"] 
    #                 + stats["memory_stats"]["stats"]["active_file"])
    #     mem_util = mem_used / 1024 / total_memory * 100
    #     print(f"{cpu_util:.3f}, {mem_util:.3f}")
