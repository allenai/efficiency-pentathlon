#include "cpu.h"
#include "mem.h"
#include "nvml.h"

// ---- SIGNAL HANDLING -------------------------------------------------------

static int interrupt = 0;

static void
signal_callback_handler(int signum)
{
  if (signum == SIGINT || signum == SIGTERM)
    interrupt = SIGTERM;
} /* signal_callback_handler */

// nvmlReturn_t nvmlDeviceGetAverageUsage(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, unsigned int* averageUsage) {
//   if (nvmlHandle == NULL) {
//     return (NVML_ERROR_LIBRARY_NOT_FOUND);
//   }

//   // We don't really use this because both the metrics we support
//   // averagePowerUsage and averageGPUUtilization are unsigned int.
//   nvmlValueType_t sampleValType;

//   // This will be set to the number of samples that can be queried. We would
//   // need to allocate an array of this size to store the samples.
//   unsigned int sampleCount;

//   // Invoking this method with `samples` set to NULL sets the sampleCount.
//   nvmlReturn_t r = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, NULL);
//   if (r != NVML_SUCCESS) {
//     return (r);
//   }

//   // Allocate memory to store sampleCount samples.
//   // In my experiments, the sampleCount at this stage was always 120 for
//   // NVML_TOTAL_POWER_SAMPLES and 100 for NVML_GPU_UTILIZATION_SAMPLES
//   nvmlSample_t* samples = (nvmlSample_t*) malloc(sampleCount * sizeof(nvmlSample_t));

//   r = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);
//   if (r != NVML_SUCCESS) {
//     free(samples);
//     return (r);
//   }

//   int i = 0;
//   unsigned int sum = 0;
//   for (; i < sampleCount; i++) {
//     sum += samples[i].sampleValue.uiVal;
//   }
//   *averageUsage = sum/sampleCount;

//   free(samples);
//   return (r);
// }

// ----------------------------------------------------------------------------

static const char * usage_msg = \
"Usage: rprof [OPTION...] command [ARG...]\n"
"Rprof -- A high frequency resources (cpu, mem, gpu, gpu mem) profiler.\n"
"\n"
"  profile_interval (ms)    Sampling profile_interval (default is 100 ms).\n"
"  output_file              Specify an output file for the collected samples.\n"
"  -b, --batch              Size of batch writing data points.\n"
"  timeout (s)             Approximate start up wait time. Increase on slow\n"
"                             machines (default is 10s).\n";

int main(int argc, char ** argv) {
  int retval = 0;
  if (argc <= 1) {
    puts(usage_msg);
    return retval;
  }
  int a      = 1;
  
  utime_t profile_interval = 100;
  if (argc >= 2){
    if (1 == strtonum(argv[1], &profile_interval)){
      printf("failed to get profile_interval, %s, use default 100 ms\n", argv[1]); 
      // return 1;
    };
  }
  printf("profile_interval=%llu ms\n", profile_interval);
  profile_interval = profile_interval * 1e3;
  FILE* output_file = stdout;
  if (argc >= 3){
    char* output_filename = argv[2];
    output_file = fopen(output_filename, "w");
    if (NULL == output_file)
    {
      printf("failed to write, %s, write to stdout\n", output_filename);
      output_file = stdout;
    }
    else {
      printf("log to %s\n", output_filename);
    }
  }

  utime_t timeout = 10; // 10 s
  if (argc >= 4){
    if (1 == strtonum(argv[3], &timeout)){
      printf("failed to get timeout, %s, default to 10 seconds\n", argv[1]); 
      // return 1;
    };
  }
  printf("timeout=%llu s\n", timeout);
  timeout = timeout*1e6;
  // Register signal handler for Ctrl+C and terminate signals.
  signal(SIGINT, signal_callback_handler);
  signal(SIGTERM, signal_callback_handler);

  // Starting profiling
  utime_t start_time = gettime();
  utime_t sample_time = gettime();

  nvmlReturn_t nv_status;
  unsigned int device_count;
  char driver_version[80];
  nv_status = nvmlInit();
  if (NVML_SUCCESS != nv_status){
    fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
    return nv_status;
  }
  nv_status = nvmlSystemGetDriverVersion(driver_version, 80);
  if (NVML_SUCCESS != nv_status){
    fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
    return nv_status;
  }
  printf("\nDriver version:  %s \n\n", driver_version);
  nv_status = nvmlDeviceGetCount(&device_count);
  if (NVML_SUCCESS != nv_status){
    fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
    return nv_status;
  }
  printf("Found %d device%s\n\n", device_count, device_count!= 1 ? "s" : "");
  printf("Listing devices:\n");
  for (unsigned i = 0; i < device_count; i++) 
  {
    nvmlDevice_t device;  
    char name[64];  
    nvmlComputeMode_t compute_mode;
    nv_status = nvmlDeviceGetHandleByIndex(i, &device);
    if (NVML_SUCCESS != nv_status){
      fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
      return nv_status;
    }
    nv_status = nvmlDeviceGetName(device, name, sizeof(name)/sizeof(name[0]));
    if (NVML_SUCCESS != nv_status){
      fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
      return nv_status;
    }
    printf("%d. %s \n", i, name);
  }
  
  unsigned long print_count = 0;
  unsigned int print_gap = 10;
  struct cpustat cpu_util_prev, cpu_util_cur;
  struct meminfo mem_util;
  get_stats(&cpu_util_prev, -1);
  usleep(profile_interval); // sleep one interval to avoid negative first sample
  fprintf(output_file, "timestamp,cpu,mem,n_gpu");
  for (unsigned device_idx = 0; device_idx < device_count; device_idx++)
  {
    fprintf(output_file, ",gpu,gpu_mem,gpu_power,gpu_clk,gpu_mem_clk");
  }
  fprintf(output_file, "\n");
  while (interrupt == 0 && (sample_time - start_time) < timeout)
  {

    get_stats(&cpu_util_cur, -1);
    double cpu_util = calculate_load(&cpu_util_prev, &cpu_util_cur);
    double mem_usage = calculate_mem_usage(&mem_util);
    
    sample_time = gettime();
    fprintf(output_file, "%.6f,%.1f,%.1f,%i", sample_time/1e6, cpu_util, mem_usage, device_count);
    if (print_count%print_gap==0)
    {
      printf("\33[2K\r");
      printf("t=%.6f, cpu=%.1f, mem=%.1f, n_gpu=%i", sample_time/1e6, cpu_util, mem_usage, device_count);
    }
    for (unsigned device_idx = 0; device_idx < device_count; device_idx++) 
    {
      nvmlDevice_t device;  
      char name[64];  
      nv_status = nvmlDeviceGetHandleByIndex(device_idx, &device);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      nvmlUtilization_t nv_util;
      nv_status = nvmlDeviceGetUtilizationRates(device, &nv_util);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      unsigned int gpu_util = nv_util.gpu;
      unsigned int gpu_mem_util = nv_util.memory;
      unsigned int gpu_power = 0;
      nv_status = nvmlDeviceGetPowerUsage(device, &gpu_power);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      unsigned int sm_clock;
      unsigned int mem_clock;

      nv_status = nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &sm_clock);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      nv_status = nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &mem_clock);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      fprintf(output_file, ",%i,%i,%i,%i,%i", gpu_util, gpu_mem_util, gpu_power, sm_clock, mem_clock);
      if (print_count%print_gap==0)
      {
        printf(", gpu=%i, gpu_mem=%i, gpu_power=%i, sm_clock=%i, gpu_mem_clock=%i", gpu_util, gpu_mem_util, gpu_power, sm_clock, mem_clock);
      }
    }
    fprintf(output_file, "\n");
    if (print_count%print_gap==0)
    {
      printf("\n");
    }
    print_count++;

    get_stats(&cpu_util_prev, -1);
    utime_t delta = gettime() - sample_time;
    if (delta < profile_interval){
      usleep(profile_interval - delta);
    }
  }

  fclose(output_file);
  nv_status = nvmlShutdown();
  if (NVML_SUCCESS != nv_status){
    fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
    return nv_status;
  }
  printf("\nelapsed %.3f ms\n", (sample_time - start_time)/1e3);
  return retval;
}
