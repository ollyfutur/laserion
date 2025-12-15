#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "temporal_profile.h"

/*
CSV to stdout:
t_fs, env

Usage:
  test_temporal --tau <fs> --tmin <fs> --tmax <fs> --dt <fs>

Example:
  ./bin/test_temporal --tau 60 --tmin -200 --tmax 200 --dt 0.5
*/

static void die_usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --tau <fs> --tmin <fs> --tmax <fs> --dt <fs>\n\n"
        "Outputs CSV to stdout:\n"
        "  t_fs,env\n",
        prog);
    exit(2);
}

static int streq(const char *a, const char *b) { return (a && b && (0 == strcmp(a,b))); }

int main(int argc, char **argv)
{
    double tau = NAN, tmin = NAN, tmax = NAN, dt = NAN;

    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--tau") && i+1 < argc) tau = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--tmin") && i+1 < argc) tmin = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--tmax") && i+1 < argc) tmax = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--dt") && i+1 < argc) dt = strtod(argv[++i], NULL);
        else die_usage(argv[0]);
    }

    if (!(tau > 0.0) || !(dt > 0.0) || !(tmax > tmin)) die_usage(argv[0]);

    GaussianTemporal gt;
    GaussianTemporal_init(&gt, tau);

    printf("t_fs,env\n");

    /* include endpoint tmax (like numpy arange-ish with explicit N) */
    long N = (long)floor((tmax - tmin) / dt + 0.5) + 1;
    if (N < 2) N = 2;

    for (long k = 0; k < N; ++k) {
        double t = tmin + (double)k * dt;
        if (t > tmax + 0.5*dt) break; /* guard */
        double env = TemporalProfile_eval((TemporalProfile *)&gt, t);
        printf("%.17g,%.17g\n", t, env);
    }

    TemporalProfile_destroy((TemporalProfile *)&gt);
    return 0;
}

