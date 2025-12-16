/* csrc/tests/test_pulse.c
 *
 * Full SinglePulse E-field comparison generator.
 * Generates N deterministic random sample points and prints CSV to stdout:
 *
 *   i,t_fs,x_um,y_um,z_um,Ex,Ey,Ez
 *
 * Sampling is done in the BEAM FRAME (x',y',z') then mapped to LAB:
 *   r_lab = r_start + x' e_xb + y' e_yb + z' k_hat
 *
 * so both C and Python evaluate exactly the same physical points.
 *
 * Usage (key options):
 *   --N <int> --seed <u64>
 *   --E0 <float> --wavelength <um> --tau <fs> --phase0 <rad>
 *   --k <kx,ky,kz> --rstart <x,y,z> --retarded 0|1
 *   --profile plane|gaussian|hermite
 *       plane:    (no extra args)
 *       gaussian: --w0 <um> --zf <um>
 *       hermite:  --w0 <um> --zf <um> --l <int> --m <int>
 *   --pol linear|circular|elliptical
 *       linear:      --angle <deg>
 *       circular:    --sense right|left --angle <deg>
 *       elliptical:  --p1 <float> --p2 <float> --delta <rad> --angle <deg>
 *
 * Sampling region (beam frame):
 *   --tspan <fs>       t in [-tspan, +tspan]
 *   --xspan <um>       x' in [-xspan, +xspan]
 *   --yspan <um>       y' in [-yspan, +yspan]
 *   --zspan <um>       z' in [zf - zspan, zf + zspan]
 *
 * Defaults:
 *   if xspan/yspan not given and profile has w0: xspan=yspan=2*w0, else 10
 *   if zspan not given and profile has w0: zspan=2*z0, else 100
 *   if tspan not given: tspan=3*tau
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

/* ---------- small helpers ---------- */

static void die_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --N <int> [--seed <u64>]\n"
            "     --E0 <float> --wavelength <um> --tau <fs> [--phase0 <rad>]\n"
            "     [--k kx,ky,kz] [--rstart x,y,z] [--retarded 0|1]\n"
            "     --profile plane|gaussian|hermite [--w0 <um> --zf <um> [--l <int> --m <int>]]\n"
            "     --pol linear|circular|elliptical [--angle <deg>] [--sense right|left]\n"
            "                              [--p1 <f> --p2 <f> --delta <rad>]\n"
            "     [--tspan <fs>] [--xspan <um>] [--yspan <um>] [--zspan <um>]\n\n"
            "Outputs CSV to stdout:\n"
            "  i,t_fs,x_um,y_um,z_um,Ex,Ey,Ez\n",
            prog);
    exit(2);
}

static int streq(const char *a, const char *b) { return (a && b && 0 == strcmp(a, b)); }

/* parse "a,b,c" into out[3] */
static int parse_vec3(const char *s, double out[3])
{
    if (!s)
        return 1;
    char *tmp = strdup(s);
    if (!tmp)
        return 2;
    char *p1 = strtok(tmp, ",");
    char *p2 = strtok(NULL, ",");
    char *p3 = strtok(NULL, ",");
    if (!p1 || !p2 || !p3)
    {
        free(tmp);
        return 3;
    }
    out[0] = strtod(p1, NULL);
    out[1] = strtod(p2, NULL);
    out[2] = strtod(p3, NULL);
    free(tmp);
    return 0;
}

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

/* uniform in [0,1) */
static double urand(void)
{
    unsigned long long r = xorshift64star();
    r >>= 11;
    return (double)r * (1.0 / 9007199254740992.0);
}

static double urand_ab(double a, double b)
{
    return a + (b - a) * urand();
}

/* ---------- beam->lab mapping: r = r_start + x' e_xb + y' e_yb + z' k_hat ---------- */
static void beam_to_lab(const SinglePulse *p,
                        double xb, double yb, double zb,
                        double r_out[3])
{
    r_out[0] = p->r_start[0] + xb * p->e_xb[0] + yb * p->e_yb[0] + zb * p->k_hat[0];
    r_out[1] = p->r_start[1] + xb * p->e_xb[1] + yb * p->e_yb[1] + zb * p->k_hat[1];
    r_out[2] = p->r_start[2] + xb * p->e_xb[2] + yb * p->e_yb[2] + zb * p->k_hat[2];
}

static double compute_z0(double w0_um, double wavelength_um)
{
    /* z0 = pi w0^2 / lambda */
    return 3.14159265358979323846 * w0_um * w0_um / wavelength_um;
}

int main(int argc, char **argv)
{
    /* required */
    long N = -1;
    unsigned long long seed = 1ull;

    double E0 = NAN;
    double wavelength_um = NAN;
    double tau_fs = NAN;
    double phase0 = 0.0;

    double k_vec[3] = {0.0, 0.0, 1.0};
    double r_start[3] = {0.0, 0.0, 0.0};
    int use_retarded_time = 1;

    const char *profile = NULL;
    double w0_um = NAN;
    double zf_um = 0.0;
    int l = 0, m = 0;

    const char *pol_kind = NULL;
    double angle_deg = 0.0;
    const char *sense = "right";
    double p1 = 1.0, p2 = 0.0, delta = 0.0;

    /* sampling */
    double tspan_fs = NAN;
    double xspan_um = NAN;
    double yspan_um = NAN;
    double zspan_um = NAN;

    for (int i = 1; i < argc; ++i)
    {
        if (streq(argv[i], "--N") && i + 1 < argc)
            N = strtol(argv[++i], NULL, 10);
        else if (streq(argv[i], "--seed") && i + 1 < argc)
            seed = (unsigned long long)strtoull(argv[++i], NULL, 10);

        else if (streq(argv[i], "--E0") && i + 1 < argc)
            E0 = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--wavelength") && i + 1 < argc)
            wavelength_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--tau") && i + 1 < argc)
            tau_fs = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--phase0") && i + 1 < argc)
            phase0 = strtod(argv[++i], NULL);

        else if (streq(argv[i], "--k") && i + 1 < argc)
        {
            if (parse_vec3(argv[++i], k_vec) != 0)
                die_usage(argv[0]);
        }
        else if (streq(argv[i], "--rstart") && i + 1 < argc)
        {
            if (parse_vec3(argv[++i], r_start) != 0)
                die_usage(argv[0]);
        }
        else if (streq(argv[i], "--retarded") && i + 1 < argc)
            use_retarded_time = atoi(argv[++i]);

        else if (streq(argv[i], "--profile") && i + 1 < argc)
            profile = argv[++i];
        else if (streq(argv[i], "--w0") && i + 1 < argc)
            w0_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--zf") && i + 1 < argc)
            zf_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--l") && i + 1 < argc)
            l = atoi(argv[++i]);
        else if (streq(argv[i], "--m") && i + 1 < argc)
            m = atoi(argv[++i]);

        else if (streq(argv[i], "--pol") && i + 1 < argc)
            pol_kind = argv[++i];
        else if (streq(argv[i], "--angle") && i + 1 < argc)
            angle_deg = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--sense") && i + 1 < argc)
            sense = argv[++i];
        else if (streq(argv[i], "--p1") && i + 1 < argc)
            p1 = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--p2") && i + 1 < argc)
            p2 = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--delta") && i + 1 < argc)
            delta = strtod(argv[++i], NULL);

        else if (streq(argv[i], "--tspan") && i + 1 < argc)
            tspan_fs = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--xspan") && i + 1 < argc)
            xspan_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--yspan") && i + 1 < argc)
            yspan_um = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--zspan") && i + 1 < argc)
            zspan_um = strtod(argv[++i], NULL);

        else
            die_usage(argv[0]);
    }

    if (!(N > 0) || !isfinite(E0) || !(wavelength_um > 0.0) || !(tau_fs > 0.0) || !profile || !pol_kind)
    {
        die_usage(argv[0]);
    }

    /* Build temporal */
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, tau_fs);

    /* Build transverse */
    PlaneWaveProfile plane;
    GaussianTransverse gauss;
    HermiteTransverse herm;

    const TransverseProfile *tp = NULL;
    int needs_w0 = 0;

    if (streq(profile, "plane"))
    {
        PlaneWaveProfile_init(&plane);
        tp = (const TransverseProfile *)&plane;
        needs_w0 = 0;
    }
    else if (streq(profile, "gaussian"))
    {
        needs_w0 = 1;
        if (!(w0_um > 0.0))
            die_usage(argv[0]);
        GaussianTransverse_init(&gauss, w0_um, zf_um);
        tp = (const TransverseProfile *)&gauss;
    }
    else if (streq(profile, "hermite"))
    {
        needs_w0 = 1;
        if (!(w0_um > 0.0))
            die_usage(argv[0]);
        HermiteTransverse_init(&herm, w0_um, zf_um, l, m);
        tp = (const TransverseProfile *)&herm;
    }
    else
    {
        die_usage(argv[0]);
    }

    /* Build polarization */
    Polarization pol;
    if (streq(pol_kind, "linear"))
    {
        if (LinearPolarization_init(&pol, k_vec, angle_deg) != 0)
        {
            fprintf(stderr, "ERROR: LinearPolarization_init failed\n");
            return 3;
        }
    }
    else if (streq(pol_kind, "circular"))
    {
        if (CircularPolarization_init(&pol, k_vec, sense, angle_deg) != 0)
        {
            fprintf(stderr, "ERROR: CircularPolarization_init failed\n");
            return 3;
        }
    }
    else if (streq(pol_kind, "elliptical"))
    {
        /* General Jones polarization on a rotated transverse basis */
        double e1[3], e2[3];

        if (Polarization_make_basis_angle(k_vec, angle_deg, e1, e2) != 0)
        {
            fprintf(stderr, "ERROR: Polarization_make_basis_angle failed for elliptical\n");
            return 3;
        }

        if (Polarization_init(&pol, e1, e2, p1, p2, delta) != 0)
        {
            fprintf(stderr, "ERROR: Polarization_init failed for elliptical\n");
            return 3;
        }
    }

    else
    {
        die_usage(argv[0]);
    }

    /* Build pulse */
    SinglePulse pulse;
    if (SinglePulse_init(&pulse,
                         E0, wavelength_um,
                         (const TemporalProfile *)&temporal,
                         tp,
                         &pol,
                         phase0,
                         k_vec,
                         use_retarded_time,
                         r_start) != 0)
    {
        fprintf(stderr, "ERROR: SinglePulse_init failed\n");
        return 4;
    }

    /* Sampling defaults */
    if (!isfinite(tspan_fs))
        tspan_fs = 3.0 * tau_fs;

    if (!isfinite(xspan_um))
        xspan_um = needs_w0 ? (2.0 * w0_um) : 10.0;
    if (!isfinite(yspan_um))
        yspan_um = needs_w0 ? (2.0 * w0_um) : 10.0;

    if (!isfinite(zspan_um))
    {
        if (needs_w0)
        {
            double z0 = compute_z0(w0_um, wavelength_um);
            zspan_um = 2.0 * z0;
        }
        else
        {
            zspan_um = 100.0;
        }
    }

    rng_state = (seed ? seed : 1ull);

    printf("i,t_fs,x_um,y_um,z_um,Ex,Ey,Ez\n");

    for (long i = 0; i < N; ++i)
    {
        double t = urand_ab(-tspan_fs, +tspan_fs);

        double xb = urand_ab(-xspan_um, +xspan_um);
        double yb = urand_ab(-yspan_um, +yspan_um);
        double zb = urand_ab(zf_um - zspan_um, zf_um + zspan_um);

        double r_lab[3];
        beam_to_lab(&pulse, xb, yb, zb, r_lab);

        double Eout[3];
        LaserPulse_E((const LaserPulse *)&pulse, t, r_lab, Eout);

        printf("%ld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
               i, t, r_lab[0], r_lab[1], r_lab[2], Eout[0], Eout[1], Eout[2]);
    }

    return 0;
}
