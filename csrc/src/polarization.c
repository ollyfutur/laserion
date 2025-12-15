#include <math.h>
#include <string.h>
#include <ctype.h>
#include "polarization.h"

/* -------------------- small vector helpers -------------------- */

static double vdot(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void vcross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static double vnorm(const double a[3])
{
    return sqrt(vdot(a, a));
}

static int vnormalize(const double in[3], double out[3])
{
    double n = vnorm(in);
    if (n <= 0.0) return 1;
    out[0] = in[0] / n;
    out[1] = in[1] / n;
    out[2] = in[2] / n;
    return 0;
}

static void vsub_scaled(double out[3], const double a[3], const double b[3], double s)
{
    /* out = a - s*b */
    out[0] = a[0] - s*b[0];
    out[1] = a[1] - s*b[1];
    out[2] = a[2] - s*b[2];
}

/* case-insensitive string compare for small tokens */
static int str_ieq(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        ++a; ++b;
    }
    return (*a == '\0' && *b == '\0');
}

/* -------------------- public API -------------------- */

int Polarization_init(Polarization *p,
                      const double e1[3],
                      const double e2[3],
                      double p1,
                      double p2,
                      double delta)
{
    if (!p) return 1;

    if (vnormalize(e1, p->e1) != 0) return 2;
    if (vnormalize(e2, p->e2) != 0) return 3;

    p->p1 = p1;
    p->p2 = p2;
    p->delta = delta;
    return 0;
}

void Polarization_vector(const Polarization *p,
                         double phase,
                         double out_vec[3])
{
    /* Re[ (p1 e1 + p2 e^{i delta} e2) e^{i phase} ]
     * = p1 e1 cos(phase) + p2 e2 cos(phase + delta)
     */
    double c1 = cos(phase);
    double c2 = cos(phase + p->delta);

    out_vec[0] = p->p1 * p->e1[0] * c1 + p->p2 * p->e2[0] * c2;
    out_vec[1] = p->p1 * p->e1[1] * c1 + p->p2 * p->e2[1] * c2;
    out_vec[2] = p->p1 * p->e1[2] * c1 + p->p2 * p->e2[2] * c2;
}

bool Polarization_is_pure_linear(const Polarization *p, double tol)
{
    return fabs(p->p2) < tol;
}

int Polarization_from_lab_direction(Polarization *p,
                                   const double k_vec[3],
                                   const double pol_hint[3],
                                   double delta)
{
    if (!p) return 1;

    /* k_hat */
    double k_hat[3];
    if (vnormalize(k_vec, k_hat) != 0) return 2;

    /* build transverse basis using z as reference if possible */
    const double z_hat[3] = {0.0, 0.0, 1.0};
    double e1[3];

    double dz = fabs(vdot(z_hat, k_hat));
    if (dz < 0.999999) {
        /* e1 = z_hat - (z_hat·k_hat) k_hat */
        vsub_scaled(e1, z_hat, k_hat, vdot(z_hat, k_hat));
        if (vnormalize(e1, e1) != 0) return 3;
    } else {
        /* fallback if k || z */
        const double x_hat[3] = {1.0, 0.0, 0.0};
        e1[0] = x_hat[0]; e1[1] = x_hat[1]; e1[2] = x_hat[2];
    }

    double e2[3];
    vcross(k_hat, e1, e2);
    if (vnormalize(e2, e2) != 0) return 4;

    /* project pol_hint onto plane perpendicular to k */
    if (vnorm(pol_hint) == 0.0) return 5;

    double ph_perp[3];
    vsub_scaled(ph_perp, pol_hint, k_hat, vdot(pol_hint, k_hat));

    if (vnorm(ph_perp) < 1e-12) {
        /* nearly parallel to k: fall back to e1 */
        ph_perp[0] = e1[0]; ph_perp[1] = e1[1]; ph_perp[2] = e1[2];
    } else {
        if (vnormalize(ph_perp, ph_perp) != 0) return 6;
    }

    /* components in (e1, e2) */
    double p1 = vdot(ph_perp, e1);
    double p2 = vdot(ph_perp, e2);

    double norm_p = hypot(p1, p2);
    if (norm_p > 0.0) {
        p1 /= norm_p;
        p2 /= norm_p;
    }

    return Polarization_init(p, e1, e2, p1, p2, delta);
}

int LinearPolarization_init(Polarization *p,
                            const double k_vec[3],
                            double angle_deg)
{
    double k_hat[3];
    if (vnormalize(k_vec, k_hat) != 0) return 2;

    const double z_hat[3] = {0.0, 0.0, 1.0};
    double e1_base[3];

    if (fabs(vdot(z_hat, k_hat)) < 0.999999) {
        vsub_scaled(e1_base, z_hat, k_hat, vdot(z_hat, k_hat));
        if (vnormalize(e1_base, e1_base) != 0) return 3;
    } else {
        const double x_hat[3] = {1.0, 0.0, 0.0};
        e1_base[0] = x_hat[0]; e1_base[1] = x_hat[1]; e1_base[2] = x_hat[2];
    }

    double e2_base[3];
    vcross(k_hat, e1_base, e2_base);
    if (vnormalize(e2_base, e2_base) != 0) return 4;

    double theta = angle_deg * (3.14159265358979323846 / 180.0);
    double p1 = cos(theta);
    double p2 = sin(theta);

    return Polarization_init(p, e1_base, e2_base, p1, p2, 0.0);
}

int CircularPolarization_init(Polarization *p,
                              const double k_vec[3],
                              const char *sense,
                              double angle_deg)
{
    double k_hat[3];
    if (vnormalize(k_vec, k_hat) != 0) return 2;

    const double z_hat[3] = {0.0, 0.0, 1.0};
    double e1_base[3];

    if (fabs(vdot(z_hat, k_hat)) < 0.999999) {
        vsub_scaled(e1_base, z_hat, k_hat, vdot(z_hat, k_hat));
        if (vnormalize(e1_base, e1_base) != 0) return 3;
    } else {
        const double x_hat[3] = {1.0, 0.0, 0.0};
        e1_base[0] = x_hat[0]; e1_base[1] = x_hat[1]; e1_base[2] = x_hat[2];
    }

    double e2_base[3];
    vcross(k_hat, e1_base, e2_base);
    if (vnormalize(e2_base, e2_base) != 0) return 4;

    /* rotate basis by angle in transverse plane */
    double theta = angle_deg * (3.14159265358979323846 / 180.0);
    double cos_t = cos(theta);
    double sin_t = sin(theta);

    double e1[3] = {
        cos_t * e1_base[0] + sin_t * e2_base[0],
        cos_t * e1_base[1] + sin_t * e2_base[1],
        cos_t * e1_base[2] + sin_t * e2_base[2]
    };

    double e2[3] = {
        -sin_t * e1_base[0] + cos_t * e2_base[0],
        -sin_t * e1_base[1] + cos_t * e2_base[1],
        -sin_t * e1_base[2] + cos_t * e2_base[2]
    };

    /* sense -> delta */
    double delta;
    if (!sense) return 5;

    if (str_ieq(sense, "right") || str_ieq(sense, "cw") || str_ieq(sense, "clockwise")) {
        delta = +3.14159265358979323846 / 2.0;
    } else if (str_ieq(sense, "left") || str_ieq(sense, "ccw") ||
               str_ieq(sense, "counterclockwise") || str_ieq(sense, "counter-clockwise")) {
        delta = -3.14159265358979323846 / 2.0;
    } else {
        return 6;
    }

    return Polarization_init(p, e1, e2, 1.0, 1.0, delta);
}

