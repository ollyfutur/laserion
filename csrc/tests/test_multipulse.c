/* csrc/tests/test_multipulse.c
 *
 * Deterministic MultiPulse comparison generator (C side).
 *
 * Builds 3 SinglePulse objects:
 *   (1) Plane wave + elliptical polarization
 *   (2) Gaussian transverse + linear polarization
 *   (3) Hermite-Gaussian HG(3,4) + circular polarization
 *
 * Then forms a MultiPulse = sum_i pulse_i.
 *
 * For each of N random (t, r_lab) points, outputs:
 *   i,t_fs,x_um,y_um,z_um,Ex,Ey,Ez,Ax,Ay,Az
 *
 * Where:
 *   E is evaluated directly from MultiPulse (sum of E_i).
 *   A is computed by numeric trapezoid integration:
 *       A(t) = -∫_{tmin}^{min(t,tmax)} E(t') dt'
 *     with uniform dt, identical logic to SinglePulse_A_impl.
 *
 * Notes:
 *   - CSV is written to stdout only.
 *   - Timings (if enabled) are written to stderr.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

/* ---------- deterministic RNG (xorshift64*) ---------- */
static unsigned long long rng_state = 88172645463325252ull;

static unsigned long long xorshift64star(void)
{
    unsigned long long x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 2685821657736338717ull;
}

static double urand01(void)
{
    unsigned long long r = xorshift64star();
    r >>= 11;
    return (double)r * (1.0/9007199254740992.0);
}

static double urand_ab(double a, double b)
{
    return a + (b - a) * urand01();
}

static int streq(const char *a, const char *b) { return (a && b && 0 == strcmp(a,b)); }

/* ---------- CLI parsing ---------- */
static void die_usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --N <int> [--seed <u64>]\n"
        "     [--tspan <fs>] [--xspan <um>] [--yspan <um>] [--zspan <um>]\n"
        "     --Atmin <fs> --Atmax <fs> --Adt <fs>\n"
        "     [--timing 0|1]\n\n"
        "Outputs CSV to stdout:\n"
        "  i,t_fs,x_um,y_um,z_um,Ex,Ey,Ez,Ax,Ay,Az\n",
        prog);
    exit(2);
}

/* ---------- A integration for any LaserPulse (MultiPulse included) ---------- */
static void pulse_A_trap(const LaserPulse *p,
                         double t_fs,
                         const double r_um[3],
                         double tmin_fs,
                         double tmax_fs,
                         double dt_fs,
                         double out_A[3])
{
    out_A[0] = out_A[1] = out_A[2] = 0.0;

    if (!(dt_fs > 0.0) || !(tmax_fs > tmin_fs)) return;

    double t_end = t_fs;
    if (t_end < tmin_fs) t_end = tmin_fs;
    if (t_end > tmax_fs) t_end = tmax_fs;

    size_t n_steps = (size_t)floor((t_end - tmin_fs) / dt_fs);

    double E_prev[3];
    LaserPulse_E(p, tmin_fs, r_um, E_prev);

    for (size_t i = 1; i <= n_steps; ++i) {
        double ti = tmin_fs + (double)i * dt_fs;
        double E_cur[3];
        LaserPulse_E(p, ti, r_um, E_cur);

        out_A[0] -= 0.5 * (E_prev[0] + E_cur[0]) * dt_fs;
        out_A[1] -= 0.5 * (E_prev[1] + E_cur[1]) * dt_fs;
        out_A[2] -= 0.5 * (E_prev[2] + E_cur[2]) * dt_fs;

        E_prev[0] = E_cur[0];
        E_prev[1] = E_cur[1];
        E_prev[2] = E_cur[2];
    }

    double t_reached = tmin_fs + (double)n_steps * dt_fs;
    double dt_last = t_end - t_reached;
    if (dt_last > 0.0) {
        double E_cur[3];
        LaserPulse_E(p, t_end, r_um, E_cur);

        out_A[0] -= 0.5 * (E_prev[0] + E_cur[0]) * dt_last;
        out_A[1] -= 0.5 * (E_prev[1] + E_cur[1]) * dt_last;
        out_A[2] -= 0.5 * (E_prev[2] + E_cur[2]) * dt_last;
    }
}

int main(int argc, char **argv)
{
    long N = -1;
    unsigned long long seed = 1ull;
    int timing = 0;

    double tspan_fs = 2000.0;
    double xspan_um = 50.0;
    double yspan_um = 50.0;
    double zspan_um = 50.0;

    double Atmin = NAN, Atmax = NAN, Adt = NAN;

    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--N") && i+1 < argc) N = strtol(argv[++i], NULL, 10);
        else if (streq(argv[i], "--seed") && i+1 < argc) seed = (unsigned long long)strtoull(argv[++i], NULL, 10);
        else if (streq(argv[i], "--tspan") && i+1 < argc) tspan_fs = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--xspan") && i+1 < argc) xspan_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--yspan") && i+1 < argc) yspan_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--zspan") && i+1 < argc) zspan_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--Atmin") && i+1 < argc) Atmin = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--Atmax") && i+1 < argc) Atmax = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--Adt") && i+1 < argc)   Adt   = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--timing") && i+1 < argc) timing = (int)strtol(argv[++i], NULL, 10);
        else die_usage(argv[0]);
    }

    if (!(N > 0) || !(isfinite(Atmin) && isfinite(Atmax) && isfinite(Adt))) die_usage(argv[0]);
    if (!(Atmax > Atmin) || !(Adt > 0.0)) die_usage(argv[0]);

    rng_state = (seed ? seed : 1ull);

    /* ---------------- build 3 pulses ---------------- */

    /* Pulse 1: plane wave + elliptical polarization */
    GaussianTemporal t1; GaussianTemporal_init(&t1, 950.0);
    PlaneWaveProfile tr1; PlaneWaveProfile_init(&tr1);
    double k1[3] = {1.0, 2.0, 3.0};
    double r1s[3] = {-25.0, 10.0, 0.0};
    Polarization pol1;
    if (LinearPolarization_init(&pol1, k1, 20.0) != 0) { fprintf(stderr, "pol1 init failed\n"); return 3; }
    pol1.p1 = 1.0; pol1.p2 = 0.55; pol1.delta = 0.7;

    SinglePulse p1;
    if (SinglePulse_init(&p1, 120.0, 10.0,
                         (const TemporalProfile *)&t1,
                         (const TransverseProfile *)&tr1,
                         &pol1,
                         0.15,
                         k1,
                         1,
                         r1s) != 0) { fprintf(stderr, "pulse1 init failed\n"); return 4; }

    /* Pulse 2: gaussian transverse + linear polarization */
    GaussianTemporal t2; GaussianTemporal_init(&t2, 780.0);
    GaussianTransverse tr2; GaussianTransverse_init(&tr2, 7.0, 60.0);
    double k2[3] = {0.6, 1.0, 1.2};
    double r2s[3] = {15.0, -20.0, 30.0};
    Polarization pol2;
    if (LinearPolarization_init(&pol2, k2, -10.0) != 0) { fprintf(stderr, "pol2 init failed\n"); return 3; }
    SinglePulse p2;
    if (SinglePulse_init(&p2, 180.0, 9.0,
                         (const TemporalProfile *)&t2,
                         (const TransverseProfile *)&tr2,
                         &pol2,
                         -0.25,
                         k2,
                         1,
                         r2s) != 0) { fprintf(stderr, "pulse2 init failed\n"); return 4; }

    /* Pulse 3: hermite HG(3,4) + circular polarization */
    GaussianTemporal t3; GaussianTemporal_init(&t3, 1100.0);
    HermiteTransverse tr3; HermiteTransverse_init(&tr3, 6.0, -40.0, 3, 4);
    double k3[3] = {1.2, 0.3, 1.0};
    double r3s[3] = {35.0, 15.0, -10.0};
    Polarization pol3;
    if (CircularPolarization_init(&pol3, k3, "right", 35.0) != 0) { fprintf(stderr, "pol3 init failed\n"); return 3; }
    SinglePulse p3;
    if (SinglePulse_init(&p3, 150.0, 10.5,
                         (const TemporalProfile *)&t3,
                         (const TransverseProfile *)&tr3,
                         &pol3,
                         0.05,
                         k3,
                         1,
                         r3s) != 0) { fprintf(stderr, "pulse3 init failed\n"); return 4; }

    const LaserPulse *arr[3] = {
        (const LaserPulse *)&p1,
        (const LaserPulse *)&p2,
        (const LaserPulse *)&p3
    };

    MultiPulse mp;
    if (MultiPulse_init(&mp, arr, 3) != 0) { fprintf(stderr, "multipulse init failed\n"); return 5; }

    const LaserPulse *pulse = (const LaserPulse *)&mp;

    printf("i,t_fs,x_um,y_um,z_um,Ex,Ey,Ez,Ax,Ay,Az\n");

    clock_t c0 = clock();

    for (long i = 0; i < N; ++i) {
        double t = urand_ab(-tspan_fs, +tspan_fs);
        double r[3] = {
            urand_ab(-xspan_um, +xspan_um),
            urand_ab(-yspan_um, +yspan_um),
            urand_ab(-zspan_um, +zspan_um)
        };

        double E[3];
        LaserPulse_E(pulse, t, r, E);

        double A[3];
        pulse_A_trap(pulse, t, r, Atmin, Atmax, Adt, A);

        printf("%ld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               i, t, r[0], r[1], r[2], E[0], E[1], E[2], A[0], A[1], A[2]);
    }

    clock_t c1 = clock();
    if (timing) {
        double elapsed = (double)(c1 - c0) / CLOCKS_PER_SEC;
        fprintf(stderr, "[C timing] multipulse: %.6f s for N=%ld\n", elapsed, N);
    }

    return 0;
}
