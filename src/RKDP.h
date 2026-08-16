#ifndef RKDP_DOT_H
#define RKDP_DOT_H
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
void RKDP_solver(double *x, double *y,
             void (*f)(double, double*, double*, void*),
                     double eps_rel, int64_t N, int64_t M,
                                         void *data, int64_t silent);
void printProgress(double fraction, time_t start_time);
#endif /* RKDP_DOT_H */
