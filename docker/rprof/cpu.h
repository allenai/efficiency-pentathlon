#ifndef CPU_H
#define CPU_H
#include "common.h"

struct cpustat {
    unsigned long t_user;
    unsigned long t_nice;
    unsigned long t_system;
    unsigned long t_idle;
    unsigned long t_iowait;
    unsigned long t_irq;
    unsigned long t_softirq;
};

void get_stats(struct cpustat *st, int cpunum)
{
    FILE *fp = fopen("/proc/stat", "r");
    int lskip = cpunum+1;
    skip_lines(fp, lskip);
    char cpun[32];
    int n = fscanf(fp, "%s %lu %lu %lu %lu %lu %lu %lu", cpun, &(st->t_user), &(st->t_nice),
        &(st->t_system), &(st->t_idle), &(st->t_iowait), &(st->t_irq),
        &(st->t_softirq));
    if (n!=8){
      fprintf(stderr, "incorrect cpu stat parsing, parsed %d\n", n);
    }
    fclose(fp);
	return;
}

void print_stats(struct cpustat *st, char *name)
{
    printf("%s: %lu %lu %lu %lu %lu %lu %lu\n", name, (st->t_user), (st->t_nice),
        (st->t_system), (st->t_idle), (st->t_iowait), (st->t_irq),
        (st->t_softirq));
}

double calculate_load(struct cpustat *prev, struct cpustat *cur)
{
    int idle_prev = (prev->t_idle) + (prev->t_iowait);
    int idle_cur = (cur->t_idle) + (cur->t_iowait);

    int nidle_prev = (prev->t_user) + (prev->t_nice) + (prev->t_system) + (prev->t_irq) + (prev->t_softirq);
    int nidle_cur = (cur->t_user) + (cur->t_nice) + (cur->t_system) + (cur->t_irq) + (cur->t_softirq);

    int total_prev = idle_prev + nidle_prev;
    int total_cur = idle_cur + nidle_cur;

    double totald = (double) total_cur - (double) total_prev;
    double idled = (double) idle_cur - (double) idle_prev;

    double cpu_perc = (1000 * (totald - idled) / totald + 1) / 10;

    return cpu_perc;
}
#endif
