#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

/*
Transverse profile test (beam frame).

Generates N random points around focus and prints CSV to stdout:
x_um,y_um,z_um,env,phase

Profiles:
  plane
  gaussian
  hermite  (requires l,m)

Common parameters:
  --profile plane|gaussian|hermite
  --wavelength <um>

Gaussian/Hermite parameters:
  --w0 <um>
  --zf <um>          (focus position along z')
  --l <int>          (hermite only)
  --m <int>          (hermite only)

Sampling region around focus:
  x in [-xspan, +xspan]
  y in [-yspan, +yspan]
  z in [zf - zspan, zf + zspan]

  --N <int>
  --seed <uint64>
  --xspan <um>       (default: 2*w0)
  --yspan <um>       (default: 2*w0)
  --zspan <um>       (default: 2*z0)

Example:
  ./bin/test_transverse --profile gaussian --wavelength 10 --w0 5 --zf 0 --N 2000 --seed 1
*/

static void die_usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --profile plane|gaussian|hermite --wavelength <um> --N <int> [--seed <u64>]\n"
        "     [--w0 <um> --zf <um> [--l <int> --m <int>]]\n"
        "     [--xspan <um> --yspan <um> --zspan <um>]\n\n"
        "Outputs CSV to stdout:\n"
        "  x_um,y_um,z_um,env,phase\n",
        prog);
    exit(2);
}

static int streq(const char *a, const char *b) { return (a && b && 0 == strcmp(a,b)); }

/* ---------- small RNG (xorshift64*) ---------- */
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
    /* 53-bit mantissa */
    unsigned long long r = xorshift64star();
    r >>= 11;
    return (double)r * (1.0/9007199254740992.0);
}

/* uniform in [a,b] */
static double urand_ab(double a, double b)
{
    return a + (b - a) * urand();
}

/* ---------- Hermite polynomials (physicists') ---------- */
/* H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1} */
static double hermite_H(int n, double x)
{
    if (n < 0) return 0.0;
    if (n == 0) return 1.0;
    if (n == 1) return 2.0*x;

    double Hnm1 = 1.0;      /* H_0 */
    double Hn   = 2.0*x;    /* H_1 */
    for (int k = 1; k < n; ++k) {
        double Hnp1 = 2.0*x*Hn - 2.0*k*Hnm1;
        Hnm1 = Hn;
        Hn = Hnp1;
    }
    return Hn;
}

/* ---------- transverse implementations ---------- */

static void compute_params(double w0, double wavelength, double *z0, double *k)
{
    *z0 = M_PI * w0*w0 / wavelength;
    *k  = 2.0 * M_PI / wavelength;
}

static double env_plane(void) { return 1.0; }
static double phase_plane(void) { return 0.0; }

static double env_hermite(double x, double y, double z,
                          double wavelength, double w0, double zf, int l, int m)
{
    double z0, k;
    compute_params(w0, wavelength, &z0, &k);
    (void)k;

    double z_rel = z - zf;
    double wz = w0 * sqrt(1.0 + (z_rel/z0)*(z_rel/z0));

    double xi = sqrt(2.0) * x / wz;
    double yi = sqrt(2.0) * y / wz;

    double Hl = hermite_H(l, xi);
    double Hm = hermite_H(m, yi);

    double gauss = exp(-(x*x + y*y) / (wz*wz));
    return (w0 / wz) * Hl * Hm * gauss;
}

static double phase_hermite(double x, double y, double z,
                            double wavelength, double w0, double zf, int l, int m)
{
    double z0, k;
    compute_params(w0, wavelength, &z0, &k);

    double z_rel = z - zf;

    double curvature = 0.0;
    if (z_rel != 0.0) {
        double R = (z_rel*z_rel + z0*z0) / z_rel;
        curvature = -k * (x*x + y*y) / (2.0 * R);
    }

    double zeta = atan2(z_rel, z0);
    double gouy = (double)(l + m + 1) * zeta;

    return curvature + gouy;
}

int main(int argc, char **argv)
{
    const char *profile = NULL;
    double wavelength = NAN;
    double w0 = NAN;
    double zf = 0.0;
    int l = 0, m = 0;

    long N = -1;
    unsigned long long seed = 1ull;

    double xspan = NAN, yspan = NAN, zspan = NAN;

    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--profile") && i+1 < argc) profile = argv[++i];
        else if (streq(argv[i], "--wavelength") && i+1 < argc) wavelength = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--w0") && i+1 < argc) w0 = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--zf") && i+1 < argc) zf = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--l") && i+1 < argc) l = (int)strtol(argv[++i], NULL, 10);
        else if (streq(argv[i], "--m") && i+1 < argc) m = (int)strtol(argv[++i], NULL, 10);
        else if (streq(argv[i], "--N") && i+1 < argc) N = strtol(argv[++i], NULL, 10);
        else if (streq(argv[i], "--seed") && i+1 < argc) seed = (unsigned long long)strtoull(argv[++i], NULL, 10);
        else if (streq(argv[i], "--xspan") && i+1 < argc) xspan = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--yspan") && i+1 < argc) yspan = strtod(argv[++i], NULL);
        else if (streq(argv[i], "--zspan") && i+1 < argc) zspan = strtod(argv[++i], NULL);
        else die_usage(argv[0]);
    }

    if (!profile || !(wavelength > 0.0) || !(N > 0)) die_usage(argv[0]);

    int is_plane   = streq(profile, "plane");
    int is_gauss   = streq(profile, "gaussian");
    int is_hermite = streq(profile, "hermite");
    if (!(is_plane || is_gauss || is_hermite)) die_usage(argv[0]);

    if (!is_plane) {
        if (!(w0 > 0.0)) die_usage(argv[0]);
        /* defaults for spans if not given */
        if (!(xspan > 0.0)) xspan = 2.0 * w0;
        if (!(yspan > 0.0)) yspan = 2.0 * w0;

        double z0, k;
        compute_params(w0, wavelength, &z0, &k);
        (void)k;
        if (!(zspan > 0.0)) zspan = 2.0 * z0;
    } else {
        if (!(xspan > 0.0)) xspan = 10.0;
        if (!(yspan > 0.0)) yspan = 10.0;
        if (!(zspan > 0.0)) zspan = 10.0;
    }

    if (is_hermite) {
        if (l < 0 || m < 0) die_usage(argv[0]);
    } else {
        l = 0; m = 0;
    }

    rng_state = (seed ? seed : 1ull);

    printf("x_um,y_um,z_um,env,phase\n");

    for (long i = 0; i < N; ++i) {
        double x = urand_ab(-xspan, +xspan);
        double y = urand_ab(-yspan, +yspan);
        double z = urand_ab(zf - zspan, zf + zspan);

        double env, ph;

        if (is_plane) {
            env = env_plane();
            ph  = phase_plane();
        } else if (is_gauss) {
            env = env_hermite(x, y, z, wavelength, w0, zf, 0, 0);
            ph  = phase_hermite(x, y, z, wavelength, w0, zf, 0, 0);
        } else { /* hermite */
            env = env_hermite(x, y, z, wavelength, w0, zf, l, m);
            ph  = phase_hermite(x, y, z, wavelength, w0, zf, l, m);
        }

        printf("%.17g,%.17g,%.17g,%.17g,%.17g\n", x, y, z, env, ph);
    }

    return 0;
}

