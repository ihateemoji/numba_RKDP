#ifndef RKDP_DOT_H
#define RKDP_DOT_H
void RKDP_solver(double *x, double *y,
             void (*f)(double, double*, double*, void*),
                     double eps_rel, int64_t N, int64_t M,
                                         void *data, int64_t silent);
#endif /* RKDP_DOT_H */
