#ifndef COMMON_H
#define COMMON_H

#include <errno.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef unsigned long long utime_t;
utime_t gettime(void) {
   struct timespec ts;
   clock_gettime(CLOCK_REALTIME, &ts);
   return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3; // microsecond
}

static int strtonum(char * str, utime_t * num) {
  char * p_err;

  *num = strtol(str, &p_err, 10);

  return  (p_err == str || *p_err != 0) ? 1 : 0;
}

void skip_lines(FILE *fp, int numlines)
{
    int cnt = 0;
    char ch;
    while((cnt < numlines) && ((ch = getc(fp)) != EOF))
    {
        if (ch == '\n')
            cnt++;
    }
    return;
}

#endif