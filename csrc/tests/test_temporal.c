#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "temporal_profile.h"

/*
 * Writes CSV with columns:
 *   t_fs, env
 *
 * Usage:
 *   ./test_temporal <tau_fs> <tmin_fs> <tmax_fs> <dt_fs> <output_csv>
 *
 * Example:
 *   ./test_temporal 60 -200 200 0.5 out_temporal_c.csv
 */

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr,
                "Usage: %s <tau_fs> <tmin_fs> <tmax_fs> <dt_fs> <output_csv>\n",
                argv[0]);
        return 2;
    }

    const double tau  = atof(argv[1]);
    const double tmin = atof(argv[2]);
    const double tmax = atof(argv[3]);
    const double dt   = atof(argv[4]);
    const char  *outf = argv[5];

    if (!(tau > 0.0) || !(tmax > tmin) || !(dt > 0.0)) {
        fprintf(stderr, "Invalid parameters.\n");
        return 3;
    }

    FILE *fp = fopen(outf, "w");
    if (!fp) {
        perror("fopen");
        return 4;
    }

    fprintf(fp, "t_fs,env\n");

    GaussianTemporal gt;
    GaussianTemporal_init(&gt, tau);

    /* Treat gt as a TemporalProfile interface */
    const TemporalProfile *tp = (const TemporalProfile *)&gt;

    /* Sample points including endpoints */
    const long N = (long)floor((tmax - tmin) / dt) + 1;

    for (long i = 0; i < N; ++i) {
        const double t = tmin + (double)i * dt;
        const double env = TemporalProfile_eval(tp, t);
        fprintf(fp, "%.17g,%.17g\n", t, env);
    }

    fclose(fp);
    return 0;
}

