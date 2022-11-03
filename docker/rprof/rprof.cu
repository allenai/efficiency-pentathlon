#include "nvml.h"
#include "cpu.h"
#include "mem.h"
#include "rprof.cuh"

extern "C" {
// ---- SIGNAL HANDLING -------------------------------------------------------

static int interrupt = 0;
static unsigned int MAX_NUM_DEVICES = 64;
static double b_to_gib = pow(2, 30);
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
"Rprof -- A high frequency resources (gpu, gpu mem) profiler.\n"
"\n"
"  profile_interval (ms)    Sampling profile_interval (default is 100 ms).\n"
"  output_file              Specify an output file for the collected samples.\n"
"  -b, --batch              Size of batch writing data points.\n"
"  timeout (s)             Approximate start up wait time. Increase on slow\n"
"                             machines (default is 10s).\n";

EXPORT int rprof(utime_t profile_interval, utime_t timeout) {
  int retval = 0;

  printf("Profile Interval: %llu ms\n", profile_interval);
  float profile_interval_in_s = profile_interval / 1e3;  // s
  profile_interval = profile_interval * 1e3;  // us
  FILE* output_file = stdout;

  output_file = fopen("workspace/log/gpu.csv", "w");
  if (NULL == output_file)
  {
    printf("Failed to write, %s, write to stdout\n", "workspace/log/gpu.csv");
    output_file = stdout;
  }
  else {
    printf("Log to %s\n", "workspace/log/gpu.csv");
  }

  printf("Timeout: %llu s\n", timeout);
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
    fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
    return nv_status;
  }
  nv_status = nvmlSystemGetDriverVersion(driver_version, 80);
  if (NVML_SUCCESS != nv_status){
    fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
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
    // nvmlComputeMode_t compute_mode;
    nv_status = nvmlDeviceGetHandleByIndex(i, &device);
    if (NVML_SUCCESS != nv_status){
      fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
      return nv_status;
    }
    nv_status = nvmlDeviceGetName(device, name, sizeof(name)/sizeof(name[0]));
    if (NVML_SUCCESS != nv_status){
      fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
      return nv_status;
    }
    printf("%d. %s \n", i, name);
  }
  
  unsigned long print_count = 0;
  unsigned int print_gap = 10;
  usleep(profile_interval); // sleep one interval to avoid negative first sample
  double energy[MAX_NUM_DEVICES] = {0.0f};  // in W.s
  unsigned long long max_mem[MAX_NUM_DEVICES] = {0}; // in Bytes
  while (interrupt == 0 && (sample_time - start_time) < timeout)
  { 
    sample_time = gettime();
    if (print_count%print_gap==0)
    {
      printf("\33[2K\r");
      printf("Timestamp: %.6f", sample_time/1e6);
    }
    for (unsigned device_idx = 0; device_idx < device_count; device_idx++) 
    {
      nvmlDevice_t device;  
      char name[64];  
      nv_status = nvmlDeviceGetHandleByIndex(device_idx, &device);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      // nvmlUtilization_t nv_util;
      // nv_status = nvmlDeviceGetUtilizationRates(device, &nv_util);
      // if (NVML_SUCCESS != nv_status){
      //   fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
      //   return nv_status;
      // }
      nvmlMemory_t memory;
      nv_status = nvmlDeviceGetMemoryInfo (device, &memory );
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      // unsigned int gpu_util = nv_util.gpu;
      unsigned long long gpu_mem = memory.used;
      // unsigned int gpu_mem_util = nv_util.memory;
      unsigned int gpu_power = 0;  // in mW
      nv_status = nvmlDeviceGetPowerUsage(device, &gpu_power);
      if (NVML_SUCCESS != nv_status){
        fprintf(stderr, "Error: %s\n", nvmlErrorString(nv_status));
        return nv_status;
      }
      float gpu_power_in_w = gpu_power / 1e3;
      energy[device_idx] = energy[device_idx] + gpu_power_in_w * profile_interval_in_s; 
      max_mem[device_idx] = max(max_mem[device_idx], gpu_mem);
      if (print_count%print_gap==0)
      {
        printf(", GPU ID: %i, GPU Memory: %llu Bytes, GPU Power: %fW", device_idx, gpu_mem, gpu_power_in_w);
      }
    }
    if (print_count%print_gap==0)
    {
      printf("\n");
    }
    print_count++;
    utime_t delta = gettime() - sample_time;
    if (delta < profile_interval){
      usleep(profile_interval - delta);
    }
  }
  float time_elapsed = (sample_time - start_time) / 1e6;
  printf("\nTime Elapsed %.3fs\n", time_elapsed);
  fprintf(output_file, "gpu_id,time_elapsed,energy,max_mem\n");
  for (unsigned device_idx = 0; device_idx < device_count; device_idx++) {
    double memory_in_gib = max_mem[device_idx] / b_to_gib;
    fprintf(output_file, "%i,%.3f,%.3f,%f\n", device_idx, time_elapsed, energy[device_idx], memory_in_gib);
    printf("GPU %i: Energy %.3ffW.s, Max Memory %f GiB\n", device_idx, energy[device_idx], memory_in_gib);
  }
  fclose(output_file);
  nv_status = nvmlShutdown();
  if (NVML_SUCCESS != nv_status){
    fprintf(stderr, "error: %s\n", nvmlErrorString(nv_status));
    return nv_status;
  }
  return retval;
}
}
