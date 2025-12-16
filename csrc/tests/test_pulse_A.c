// csrc/tests/test_pulse_A.c
//
// Generate random sample points and compute vector potential A(t,r) for a SinglePulse.
// Output CSV to stdout (no header):
//   i, pos_id, t, x, y, z, Ax, Ay, Az
//
// Sampling strategy:
//   - Generate Npos spatial positions r_lab (uniform in a box).
//   - For each position, generate Nt times t (uniform in [-tspan/2, +tspan/2]).
//   - Total rows = Npos * Nt.
//
// Intended consumer: tests/compare_pulse_A.py

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

static double urand01(void) { return (double)rand() / (double)RAND_MAX; }
static double urand_sym(double span) { return (urand01() - 0.5) * span; }

static int parse_vec3(const char *s, double out[3])
{
    if (!s)
        return 1;
    double a, b, c;
    if (sscanf(s, "%lf,%lf,%lf", &a, &b, &c) != 3)
        return 2;
    out[0] = a;
    out[1] = b;
    out[2] = c;
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s \\\n"
            "  --Npos N --Nt N --seed S \\\n"
            "  --E0 E0 --wavelength lam --tau tau --phase0 ph0 \\\n"
            "  --k kx,ky,kz --rstart x0,y0,z0 --retarded 0|1 \\\n"
            "  --profile plane|gaussian|hermite [--w0 w0 --zf zf --l l --m m] \\\n"
            "  --pol linear|circular|elliptical [--angle deg] [--sense left|right] [--p1 v --p2 v --delta rad] \\\n"
            "  --Atmin tmin --Atmax tmax --Adt dt \\\n"
            "  [--tspan T] [--xspan X] [--yspan Y] [--zspan Z]\n",
            prog);
}

int main(int argc, char **argv)
{
    int Npos = 40;
    int Nt = 100;
    int seed = 1;

    double E0 = 150.0;
    double wavelength = 10.0;
    double tau = 1000.0;
    double phase0 = 0.0;

    double k_vec[3] = {0.0, 0.0, 1.0};
    double r_start[3] = {0.0, 0.0, 0.0};
    int retarded = 1;

    const char *profile = "plane";
    double w0 = 5.0, zf = 0.0;
    int l = 0, m = 0;

    const char *pol = "linear";
    double angle_deg = 0.0;
    const char *sense = "right";
    double p1 = 1.0, p2 = 0.0, delta = 0.0;

    // Sampling spans (lab frame)
    double tspan = -1.0; // if <0, use 4*tau
    double xspan = 20.0;
    double yspan = 20.0;
    double zspan = 200.0;

    // A integration window
    double Atmin = -4000.0;
    double Atmax = +4000.0;
    double Adt = 0.01;

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--Npos") && i + 1 < argc)
            Npos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--Nt") && i + 1 < argc)
            Nt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);

        else if (!strcmp(argv[i], "--E0") && i + 1 < argc)
            E0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wavelength") && i + 1 < argc)
            wavelength = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tau") && i + 1 < argc)
            tau = atof(argv[++i]);
        else if (!strcmp(argv[i], "--phase0") && i + 1 < argc)
            phase0 = atof(argv[++i]);

        else if (!strcmp(argv[i], "--k") && i + 1 < argc)
        {
            if (parse_vec3(argv[++i], k_vec))
                return 2;
        }
        else if (!strcmp(argv[i], "--rstart") && i + 1 < argc)
        {
            if (parse_vec3(argv[++i], r_start))
                return 3;
        }
        else if (!strcmp(argv[i], "--retarded") && i + 1 < argc)
            retarded = atoi(argv[++i]);

        else if (!strcmp(argv[i], "--profile") && i + 1 < argc)
            profile = argv[++i];
        else if (!strcmp(argv[i], "--w0") && i + 1 < argc)
            w0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--zf") && i + 1 < argc)
            zf = atof(argv[++i]);
        else if (!strcmp(argv[i], "--l") && i + 1 < argc)
            l = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m") && i + 1 < argc)
            m = atoi(argv[++i]);

        else if (!strcmp(argv[i], "--pol") && i + 1 < argc)
            pol = argv[++i];
        else if (!strcmp(argv[i], "--angle") && i + 1 < argc)
            angle_deg = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sense") && i + 1 < argc)
            sense = argv[++i];
        else if (!strcmp(argv[i], "--p1") && i + 1 < argc)
            p1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--p2") && i + 1 < argc)
            p2 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--delta") && i + 1 < argc)
            delta = atof(argv[++i]);

        else if (!strcmp(argv[i], "--tspan") && i + 1 < argc)
            tspan = atof(argv[++i]);
        else if (!strcmp(argv[i], "--xspan") && i + 1 < argc)
            xspan = atof(argv[++i]);
        else if (!strcmp(argv[i], "--yspan") && i + 1 < argc)
            yspan = atof(argv[++i]);
        else if (!strcmp(argv[i], "--zspan") && i + 1 < argc)
            zspan = atof(argv[++i]);

        else if (!strcmp(argv[i], "--Atmin") && i + 1 < argc)
            Atmin = atof(argv[++i]);
        else if (!strcmp(argv[i], "--Atmax") && i + 1 < argc)
            Atmax = atof(argv[++i]);
        else if (!strcmp(argv[i], "--Adt") && i + 1 < argc)
            Adt = atof(argv[++i]);

        else if (!strcmp(argv[i], "--help"))
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (Npos <= 0 || Nt <= 0)
    {
        fprintf(stderr, "Npos and Nt must be positive.\n");
        return 4;
    }
    if (tspan < 0.0)
        tspan = 4.0 * tau;

    srand((unsigned)seed);

    // --- build temporal ---
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, tau);

    // --- build transverse ---
    PlaneWaveProfile plane;
    GaussianTransverse gauss;
    HermiteTransverse herm;

    const TransverseProfile *tp = NULL;
    if (!strcmp(profile, "plane"))
    {
        PlaneWaveProfile_init(&plane);
        tp = &plane.base;
    }
    else if (!strcmp(profile, "gaussian"))
    {
        GaussianTransverse_init(&gauss, w0, zf);
        tp = &gauss.base;
    }
    else if (!strcmp(profile, "hermite"))
    {
        HermiteTransverse_init(&herm, w0, zf, l, m);
        tp = &herm.base;
    }
    else
    {
        fprintf(stderr, "Unknown profile: %s\n", profile);
        return 5;
    }

    // --- build polarization using general Jones-based Polarization ---
    Polarization P;

    // replicate the same transverse basis logic as SinglePulse_init + rotate by angle
    double kn = sqrt(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]);
    if (kn <= 0.0)
    {
        fprintf(stderr, "Invalid k vector.\n");
        return 6;
    }
    double k_hat[3] = {k_vec[0] / kn, k_vec[1] / kn, k_vec[2] / kn};

    double zhat[3] = {0, 0, 1};
    double zdot = zhat[0] * k_hat[0] + zhat[1] * k_hat[1] + zhat[2] * k_hat[2];

    double e1[3];
    if (fabs(zdot) < 0.999999)
    {
        e1[0] = zhat[0] - zdot * k_hat[0];
        e1[1] = zhat[1] - zdot * k_hat[1];
        e1[2] = zhat[2] - zdot * k_hat[2];
        double n = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
        e1[0] /= n;
        e1[1] /= n;
        e1[2] /= n;
    }
    else
    {
        e1[0] = 1;
        e1[1] = 0;
        e1[2] = 0;
    }

    double e2[3] = {
        k_hat[1] * e1[2] - k_hat[2] * e1[1],
        k_hat[2] * e1[0] - k_hat[0] * e1[2],
        k_hat[0] * e1[1] - k_hat[1] * e1[0]};
    {
        double n = sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
        if (n <= 0.0)
        {
            fprintf(stderr, "Invalid transverse basis.\n");
            return 7;
        }
        e2[0] /= n;
        e2[1] /= n;
        e2[2] /= n;
    }

    double th = angle_deg * (3.14159265358979323846 / 180.0);
    double c = cos(th), s = sin(th);

    double e1r[3] = {c * e1[0] + s * e2[0], c * e1[1] + s * e2[1], c * e1[2] + s * e2[2]};
    double e2r[3] = {-s * e1[0] + c * e2[0], -s * e1[1] + c * e2[1], -s * e1[2] + c * e2[2]};

    double p2_use = p2;
    double delta_use = delta;

    if (!strcmp(pol, "linear"))
    {
        p1 = 1.0;
        p2_use = 0.0;
        delta_use = 0.0;
    }
    else if (!strcmp(pol, "circular"))
    {
        p1 = 1.0;
        p2_use = 1.0;
        delta_use = +3.14159265358979323846 / 2.0;
        if (!strcmp(sense, "left"))
            delta_use = -3.14159265358979323846 / 2.0;
    }
    else if (!strcmp(pol, "elliptical"))
    {
        // keep p1, p2_use, delta_use as provided
    }
    else
    {
        fprintf(stderr, "Unknown polarization: %s\n", pol);
        return 8;
    }

    Polarization_init(&P, e1r, e2r, p1, p2_use, delta_use);

    // --- build pulse ---
    SinglePulse pulse;
    if (SinglePulse_init(&pulse, E0, wavelength, &temporal.base, tp, &P, phase0, k_vec, retarded, r_start) != 0)
    {
        fprintf(stderr, "SinglePulse_init failed.\n");
        return 9;
    }
    SinglePulse_enable_A(&pulse, Atmin, Atmax, Adt);

    // --- generate positions ---
    double *pos_xyz = (double *)malloc((size_t)Npos * 3 * sizeof(double));
    if (!pos_xyz)
    {
        fprintf(stderr, "alloc failed\n");
        return 10;
    }

    for (int pidx = 0; pidx < Npos; ++pidx)
    {
        pos_xyz[3 * pidx + 0] = r_start[0] + urand_sym(xspan);
        pos_xyz[3 * pidx + 1] = r_start[1] + urand_sym(yspan);
        pos_xyz[3 * pidx + 2] = r_start[2] + urand_sym(zspan);
    }

    // --- output ---
    long long row = 0;
    clock_t t_start = clock();
    for (int pidx = 0; pidx < Npos; ++pidx)
    {
        double r_lab[3] = {pos_xyz[3 * pidx + 0], pos_xyz[3 * pidx + 1], pos_xyz[3 * pidx + 2]};

        for (int tidx = 0; tidx < Nt; ++tidx)
        {
            double t = urand_sym(tspan);
            double A[3];
            LaserPulse_A(&pulse.base, t, r_lab, A);

            printf("%lld,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                   row, pidx, t, r_lab[0], r_lab[1], r_lab[2], A[0], A[1], A[2]);
            row++;
        }
    }

    clock_t t_end = clock();
    double elapsed_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    fprintf(stderr,
            "[C timing] LaserPulse_A: %.6f s for %d positions × %d times (%lld calls)\n",
            elapsed_sec, Npos, Nt, (long long)Npos * (long long)Nt);

    free(pos_xyz);
    return 0;
}
