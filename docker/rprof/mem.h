#ifndef MEM_H
#define MEM_H
#include "common.h"

struct meminfo {
    unsigned long t_total;
    unsigned long t_free;
    unsigned long t_available;
};

double calculate_mem_usage(struct meminfo *mt){
    FILE *fp = fopen("/proc/meminfo", "r");
    int n = fscanf(fp, "MemTotal:\t%lu kB\nMemFree:\t%lu kB\nMemAvailable:\t%lu kB\n", &(mt->t_total),&(mt->t_free),&(mt->t_available));
    if (n!=3){
      fprintf(stderr, "incorrect meminfo parsing, parsed %d\n", n);
    }
    fclose(fp);
  return  (1000 * (mt->t_total - mt->t_available) / mt->t_total + 1) / 10;
}

//
///* Parse the contents of /proc/meminfo (in buf), return value of "name"
// * (example: "MemTotal:")
// * Returns -errno if the entry cannot be found. */
//static long long get_entry(const char* name, const char* buf)
//{
//    char* hit = strstr(buf, name);
//    if (hit == NULL) {
//        return -ENODATA;
//    }
//
//    errno = 0;
//    long long val = strtoll(hit + strlen(name), NULL, 10);
//    if (errno != 0) {
//        int strtoll_errno = errno;
//        fprintf(stderr, "%s: strtol() failed: %s", __func__, strerror(errno));
//        return -strtoll_errno;
//    }
//    return val;
//}
//
///* Like get_entry(), but exit if the value cannot be found */
//static long long get_entry_fatal(const char* name, const char* buf)
//{
//    long long val = get_entry(name, buf);
//    if (val < 0) {
//        // warn("%s: fatal error, dumping buffer for later diagnosis:\n%s", __func__, buf);
//        fprintf(stderr, "could not find entry '%s' in /proc/meminfo: %s\n", name, strerror((int)-val));
//    }
//    return val;
//}

/* Parse /proc/meminfo.
 * This function either returns valid data or kills the process
 * with a fatal error.
 */
//double calculate_mem_usage()
//{
//    // Note that we do not need to close static FDs that we ensure to
//    // `fopen()` maximally once.
//    static FILE* fd;
//    // On Linux 5.3, "wc -c /proc/meminfo" counts 1391 bytes.
//    // 2048 should be enough for the foreseeable future.
//    char buf[2048] = { 0 };
//    if (fd == NULL)
//        fd = fopen("/proc/meminfo", "r");
//    if (fd == NULL) {
//        fprintf(stderr, "could not open /proc/meminfo: %s\n", strerror(errno));
//    }
//    rewind(fd);
//
//    size_t len = fread(buf, 1, sizeof(buf) - 1, fd);
//    if (ferror(fd)) {
//        fprintf(stderr, "could not read /proc/meminfo: %s\n", strerror(errno));
//    }
//    if (len == 0) {
//        fprintf(stderr, "could not read /proc/meminfo: 0 bytes returned\n");
//    }
//
//    long long mem_total = get_entry_fatal("MemTotal:", buf);
//    long long mem_avail = get_entry_fatal("MemAvailable:", buf);
//    // printf("mem util: %lld, %lld\n", mem_total, mem_avail);
////    fclose(fd);
//    // Calculate percentages
//    return  (1000 * (mem_total - mem_avail) / mem_total + 1) / 10;
//}

#endif
