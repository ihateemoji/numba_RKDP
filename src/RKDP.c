#include"RKDP.h"

/*                   ARRAY MANIPULATION BLOCK                                */
void *xmalloc(size_t bytes) {
    /* Lazy malloc wrapper that terminates the execution upon memory allocation
            failure. In principle, this one is problematic as it does not
            free any other memory that may have been already allocated.
            However, if call to malloc fails, we probably have bigger
            problems ...
        Input:
            <size_t> - number of bytes to allocate
        Returns:
            <*void>  - pointer to the allocated memory */
    void *ptr;
    if ( (ptr = malloc(bytes)) == NULL )
    {
        fprintf(stderr, "Failed to allocate memory, aborting...\n");
        exit(EXIT_FAILURE);
    }
    return (ptr);
}

void cp_array(double *x, double *y, int64_t N) {
    /* Copies array
       Inputs:
        <*double> - pointer to the first element of the array to copy
        <*double> - pointer to the first element of the destination array
        <int>     - number of elements */
    #pragma omp simd
    for (int64_t mu = 0; mu < N; mu++) {
        *(y+mu) = *(x+mu);
    }
}

/*                     MATHEMATICAL FUNCTIONS BLOCK                          */
double min(double x1, double x2) {
    /* Simple function that computes a minimum of two doubles
        Inputs:
            <double> - first number
            <double> - second number
        Returns:
            <double> - minima of two inputs */
    if (x1 < x2) { return x1; }
    return x2;
}

void Adaptive_Step(void (*f)(double, double*, double*, void*),
                double x0, double x1, double *y0, double *y1,
                double *hin, double eps_in, double *temp, double *temp_y,
                double *K1, double *K2, double *K3, double *K4,
                double *K5, double *K6, double *K7, int64_t N, void *data) {
    /* Procedure that adaptively advances the solution to the system of ODEs.
        Inputs:
            <void(double, *double, *double, *void)> -
                        procedure corresponding to the right hand side of
                                                            the system of ODEs,
                            first input -> current value of x
                            second input -> current solution vector
                            third input -> empty vector to store the
                                                numerical value of the RHS in
                            fourth input -> any data passed through
                                                            to the function
            <double>  - starting position
            <double>  - position to advance to
            <*double> - pointer to the solution at the starting position
            <*double> - pointer to the solution at the advanced position
                                    (will be overwritten)
            <*double> - pointer to the initial step-size guess
            <double>  - target numerical tolerance of the solution
            <*double> - pointer to the array used for the temporary solution
            <*double> - pointer to the second array used for
                                                the temporary solution
            <*double> - pointer to the array to store first RK coefficients
            <*double> - pointer to the array to store second RK coefficients
            <*double> - pointer to the array to store third RK coefficients
            <*double> - pointer to the array to store fourth RK coefficients
            <*double> - pointer to the array to store fith RK coefficients
            <*double> - pointer to the array to store sixth RK coefficients
            <*double> - pointer to the array to store seventh RK coefficients
            <int>     - dimensionality of the problem */
    double x = x0; /* starting position */
    double TE = 0.0; /* varuable to store trucnation error */
    double total_y = 0.0; /* variable to store the magnitude
                                                    of solution vector */
    double h = *hin; /* initial estimate of the stepsize */
    cp_array(y0, temp_y, N); /* store current solution */
    /* we advance the system untill the value of x1 is reached */
    f(x, temp_y, K1, data); /* only need to compute K1 once */
    do {
        /* we iteratively reduce the step and compute the truncation
                     error untill the desired relative error is achieved */
        do {
            /* computation of the RK coefficients as per the
                Butcher table from
            https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method */
            #pragma omp simd
            for (int64_t nu = 0; nu < N; nu++) {
                *(temp+nu) = *(temp_y+nu) + h * 1.0/5.0 * *(K1+nu);
            }
            f(x+1.0/5.0*h, temp, K2, data);
            #pragma omp simd
            for (int64_t nu = 0; nu < N; nu++) {
                *(temp+nu) = *(temp_y+nu) + h * 3.0/40.0 * *(K1+nu)
                                          + h * 9.0/40.0 * *(K2+nu);
            }
            f(x+3.0/10.0*h, temp, K3, data);
            #pragma omp simd
            for (int64_t nu = 0; nu < N; nu++) {
                *(temp+nu) = *(temp_y+nu) + h * 44.0/45.0 * *(K1+nu)
                                          - h * 56.0/15.0 * *(K2+nu)
                                          + h * 32.0/9.0  * *(K3+nu);
            }
            f(x+4.0/5.0*h, temp, K4, data);
            #pragma omp simd
            for (int64_t nu = 0; nu < N; nu++) {
                *(temp+nu) = *(temp_y+nu) + h * 19372.0/6561.0 * *(K1+nu)
                                          - h * 25360.0/2187.0 * *(K2+nu)
                                          + h * 64448.0/6561.0 * *(K3+nu)
                                          - h *   212.0/729.0  * *(K4+nu);
            }
            f(x+8.0/9.0*h, temp, K5, data);
            #pragma omp simd
            for (int64_t nu = 0; nu < N; nu++) {
                *(temp+nu) = *(temp_y+nu) + h *  9017.0/3168.0 * *(K1+nu)
                                          - h *   355.0/33.0   * *(K2+nu)
                                          + h * 46732.0/5247.0 * *(K3+nu)
                                          + h *   49.0/176.0   * *(K4+nu)
                                          - h * 5103.0/18656.0 * *(K5+nu);
            }
            f(x+h, temp, K6, data);
            #pragma omp simd
            for (int64_t nu = 0; nu < N; nu++) {
                *(temp+nu) = *(temp_y+nu) + h *   35.0/384.0  * *(K1+nu)
                                          + h *  500.0/1113.0 * *(K3+nu)
                                          + h *  125.0/192.0  * *(K4+nu)
                                          - h * 2187.0/6784.0 * *(K5+nu)
                                          + h *   11.0/84.0   * *(K6+nu);
            }
            f(x+h, temp, K7, data);
            /* compute relative truncation error */
            TE = 0.0;
            total_y = 0.0;
            for (int64_t nu = 0; nu < N; nu++) {
                TE += pow( (  35.0/384.0  -  5179.0/57600.0)   * *(K1+nu) * h
                         + ( 500.0/1113.0 -  7571.0/16695.0)   * *(K3+nu) * h
                         + ( 125.0/192.0  -   393.0/640.0)     * *(K4+nu) * h
                         - (2187.0/6784.0 - 92097.0/339200.0)  * *(K5+nu) * h
                         + (  11.0/84.0   -  187.0/2100.0)     * *(K6+nu) * h
                         -     1.0/40.0 * *(K7+nu) * h, 2.0);
                total_y += pow( *(temp+nu), 2.0);
            }
            TE = sqrt(TE);
            total_y = sqrt(total_y);
            /* if the error tolerance is not achieved we decrease
                                                                    step */
            if (TE > eps_in*total_y) {
                /* note that we take special care for the case where
                                                        x is close to x1 */
                h = min(0.9 * h * pow(eps_in/TE*total_y, 1.0/5.0), x1 - x);
            }
        } while (TE > eps_in*total_y);
        /* once the error threshold is achieved, we advance the solution */
        cp_array(K7, K1, N); /* use FSAL property of the method */
        #pragma omp simd
        for (int64_t nu = 0; nu < N; nu++) {
            /* local extrapolation -> use higher-order solution to
                                                continue the integration */
            *(temp_y+nu) = *(temp_y+nu) +  5179.0/57600.0  * *(K1+nu) * h
                                        +  7571.0/16695.0  * *(K3+nu) * h
                                        +   393.0/640.0    * *(K4+nu) * h
                                        - 92097.0/339200.0 * *(K5+nu) * h
                                        +   187.0/2100.0   * *(K6+nu) * h
                                        +     1.0/40.0     * *(K7+nu) * h;
        }
        /* advance x */
        x += h;
        /* overwrite the initial guess variable for use on the next step */
        *hin = h;
        /* increase the stepsize if possible */
        h = min(0.9 * h * pow(eps_in/TE*total_y, 1.0/5.0), x1 - x);
    } while (x < x1);
    /* once the solution is reached we store it in y1 */
    cp_array(temp_y, y1, N);
}

void RKDP_solver(double *x, double *y,
            void (*f)(double, double*, double*, void*),
                    double eps_rel, int64_t N, int64_t M,
                                        void *data, int64_t silent) {
    /* Adaptive Dormand–Prince solver of the system of ODEs.
        The ODE is solved on the provided x domain. However, the number of
        steps between discrete points of x domain is chosen adaptively
        to satisfy the provided relative error condition.
        Inputs:
            <*double> - x domain
            <*double> - y (N*M) array to store the solution in,
                            *(y+i) elements determine the initial condition
            <void(double, *double, *double, *void)> -
                        procedure corresponding to the right hand side of
                                                            the system of ODEs,
                            first input -> current value of x
                            second input -> current solution vector
                            third input -> empty vector to store the
                                                numerical value of the RHS in
                            fourth input -> any data passed through
                                                            to the function
            <double>  - target relative error of the solution
            <int>     - dimensionality of the problem
            <int>     - number of points in x domain
            <*void>   - any data passed through to the function
            <int>     - flag indicating weather or not to print the status
                            (0 -> print the status)
                            (1 -> suppress status printing) */
    /* initial memory allocation */
    double *temp = xmalloc(N*sizeof(double));
    double *temp_y = xmalloc(N*sizeof(double));
    double *K1 = xmalloc(N*sizeof(double));
    double *K2 = xmalloc(N*sizeof(double));
    double *K3 = xmalloc(N*sizeof(double));
    double *K4 = xmalloc(N*sizeof(double));
    double *K5 = xmalloc(N*sizeof(double));
    double *K6 = xmalloc(N*sizeof(double));
    double *K7 = xmalloc(N*sizeof(double));
    /* initialise vector to zero */
    for (int64_t mu = 0; mu < N; mu++) {
        *(temp+mu) = 0.0;
        *(temp_y+mu) = 0.0;
        *(K1+mu) = 0.0;
        *(K2+mu) = 0.0;
        *(K3+mu) = 0.0;
        *(K4+mu) = 0.0;
        *(K5+mu) = 0.0;
        *(K6+mu) = 0.0;
        *(K7+mu) = 0.0;
    }
    /* iterate the solver */
    double h = *(x+1) - *x; /* initial stepsize guess */
    int64_t Niter = M-1;
    if (silent == 0) {
        printf("Performing RKDP evolution for %"PRId64" steps!\n", Niter);
    }
    time_t start_time;
    time(&start_time);
    for (int64_t mu = 1; mu < M; mu++) {
        /* take adaptive steps and update the stepsize guess */
        Adaptive_Step(f, *(x+mu-1), *(x+mu), (y+N*(mu-1)), (y+N*mu), &h,
                   eps_rel, temp, temp_y, K1, K2, K3, K4, K5, K6, K7, N, data);
        if (silent == 0) {
            printProgress(((double) mu)/((double) Niter), start_time);
        }
    }
    if (silent == 0) {
        printf("\n");
    }
    /* free the memory */
    free(temp);
    free(temp_y);
    free(K1);
    free(K2);
    free(K3);
    free(K4);
    free(K5);
    free(K6);
    free(K7);
}

/*                              IO BLOCK                                     */
void printProgress(double fraction, time_t start_time) {
    /* Basic progress bar implementation.
        Input:
            <double> - fraction of the task done
            <time_t> - start time of the task execution */
    /* compute the percentage value of completed tasks */
    int val = (int) (fraction * 100);
    int lpad = (int) (fraction * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    /* compute elapsed time */
    time_t now;
    time(&now);
    double elapsed = difftime(now, start_time);
    /* estimate remaining time */
    double estimated_total_time = elapsed / fraction;
    double remaining_time = estimated_total_time - elapsed;
    /* convert time to minutes and seconds */
    int elapsed_minutes = (int) elapsed / 60;
    int elapsed_seconds = (int) elapsed % 60;
    int remaining_minutes = (int) remaining_time / 60;
    int remaining_seconds = (int) remaining_time % 60;
    /* print progress bar */
    printf("\r%3d%% [%.*s%*s] [%02d:%02d<%02d:%02d]",
            val, lpad, PBSTR, rpad, "", elapsed_minutes, elapsed_seconds,
                                        remaining_minutes, remaining_seconds);
    fflush(stdout);
}
