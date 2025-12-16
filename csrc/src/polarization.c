/* csrc/src/polarization.c */
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "polarization.h"

/* -------------------- small vector helpers -------------------- */

static double vdot(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void vcross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static double vnorm(const double a[3])
{
    return sqrt(vdot(a, a));
}

static int vnormalize(const double in[3], double out[3])
{
    double n = vnorm(in);
    if (n <= 0.0)
        return 1;
    out[0] = in[0] / n;
    out[1] = in[1] / n;
    out[2] = in[2] / n;
    return 0;
}

static void vsub_scaled(double out[3], const double a[3], const double b[3], double s)
{
    /* out = a - s*b */
    out[0] = a[0] - s * b[0];
    out[1] = a[1] - s * b[1];
    out[2] = a[2] - s * b[2];
}

/* case-insensitive string compare for small tokens */
static int str_ieq(const char *a, const char *b)
{
    while (*a && *b)
    {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        ++a;
        ++b;
    }
    return (*a == '\0' && *b == '\0');
}

/* Build lab-tied transverse basis (e1_base, e2_base) for a given k_hat. */
static int build_transverse_basis(const double k_hat[3],
                                  double e1_base[3],
                                  double e2_base[3])
{
    const double z_hat[3] = {0.0, 0.0, 1.0};

    if (fabs(vdot(z_hat, k_hat)) < 0.999999)
    {
        /* e1_base = z_hat - (z_hat·k_hat) k_hat */
        vsub_scaled(e1_base, z_hat, k_hat, vdot(z_hat, k_hat));
        if (vnormalize(e1_base, e1_base) != 0)
            return 3;
    }
    else
    {
        /* fallback if k || z */
        const double x_hat[3] = {1.0, 0.0, 0.0};
        e1_base[0] = x_hat[0];
        e1_base[1] = x_hat[1];
        e1_base[2] = x_hat[2];
    }

    vcross(k_hat, e1_base, e2_base);
    if (vnormalize(e2_base, e2_base) != 0)
        return 4;

    return 0;
}

/* Rotate (e1_base,e2_base) by angle_deg in the transverse plane. */
static void rotate_basis(const double e1_base[3],
                         const double e2_base[3],
                         double angle_deg,
                         double e1[3],
                         double e2[3])
{
    const double pi = 3.14159265358979323846;
    double theta = angle_deg * (pi / 180.0);
    double c = cos(theta);
    double s = sin(theta);

    /* e1 =  c e1_base + s e2_base
       e2 = -s e1_base + c e2_base */
    e1[0] = c * e1_base[0] + s * e2_base[0];
    e1[1] = c * e1_base[1] + s * e2_base[1];
    e1[2] = c * e1_base[2] + s * e2_base[2];

    e2[0] = -s * e1_base[0] + c * e2_base[0];
    e2[1] = -s * e1_base[1] + c * e2_base[1];
    e2[2] = -s * e1_base[2] + c * e2_base[2];
}

/* sense -> delta (+pi/2 right-handed, -pi/2 left-handed) */
static int sense_to_delta(const char *sense, double *delta_out)
{
    const double pi = 3.14159265358979323846;
    if (!sense || !delta_out)
        return 1;

    if (str_ieq(sense, "right") || str_ieq(sense, "cw") || str_ieq(sense, "clockwise"))
    {
        *delta_out = +pi / 2.0;
        return 0;
    }
    if (str_ieq(sense, "left") || str_ieq(sense, "ccw") ||
        str_ieq(sense, "counterclockwise") || str_ieq(sense, "counter-clockwise"))
    {
        *delta_out = -pi / 2.0;
        return 0;
    }
    return 2;
}

/* -------------------- public API -------------------- */

int Polarization_init(Polarization *p,
                      const double e1[3],
                      const double e2[3],
                      double p1,
                      double p2,
                      double delta)
{
    if (!p)
        return 1;

    if (vnormalize(e1, p->e1) != 0)
        return 2;
    if (vnormalize(e2, p->e2) != 0)
        return 3;

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

int Polarization_make_basis_angle(const double k_vec[3],
                                  double angle_deg,
                                  double e1_out[3],
                                  double e2_out[3])
{
    if (!k_vec || !e1_out || !e2_out)
        return 1;

    /* k_hat */
    double k_hat[3];
    if (vnormalize(k_vec, k_hat) != 0)
        return 2;

    double e1_base[3], e2_base[3];
    {
        int rc = build_transverse_basis(k_hat, e1_base, e2_base);
        if (rc != 0)
            return rc;
    }

    rotate_basis(e1_base, e2_base, angle_deg, e1_out, e2_out);
    return 0;
}

int Polarization_from_lab_direction(Polarization *p,
                                    const double k_vec[3],
                                    const double pol_hint[3],
                                    double delta)
{
    if (!p)
        return 1;

    /* k_hat */
    double k_hat[3];
    if (vnormalize(k_vec, k_hat) != 0)
        return 2;

    /* base transverse basis tied to lab */
    double e1_base[3], e2_base[3];
    {
        int rc = build_transverse_basis(k_hat, e1_base, e2_base);
        if (rc != 0)
            return rc;
    }

    /* project pol_hint onto plane perpendicular to k */
    if (vnorm(pol_hint) == 0.0)
        return 5;

    double ph_perp[3];
    vsub_scaled(ph_perp, pol_hint, k_hat, vdot(pol_hint, k_hat));

    if (vnorm(ph_perp) < 1e-12)
    {
        /* nearly parallel to k: fall back to e1_base */
        ph_perp[0] = e1_base[0];
        ph_perp[1] = e1_base[1];
        ph_perp[2] = e1_base[2];
    }
    else
    {
        if (vnormalize(ph_perp, ph_perp) != 0)
            return 6;
    }

    /* components in (e1_base, e2_base) */
    double p1 = vdot(ph_perp, e1_base);
    double p2 = vdot(ph_perp, e2_base);

    double norm_p = hypot(p1, p2);
    if (norm_p > 0.0)
    {
        p1 /= norm_p;
        p2 /= norm_p;
    }

    return Polarization_init(p, e1_base, e2_base, p1, p2, delta);
}

int LinearPolarization_init(Polarization *p,
                            const double k_vec[3],
                            double angle_deg)
{
    if (!p)
        return 1;

    double e1[3], e2[3];
    int rc = Polarization_make_basis_angle(k_vec, angle_deg, e1, e2);
    if (rc != 0)
        return rc;

    /* linear along e1 (after rotation) */
    return Polarization_init(p, e1, e2, 1.0, 0.0, 0.0);
}

int CircularPolarization_init(Polarization *p,
                              const double k_vec[3],
                              const char *sense,
                              double angle_deg)
{
    if (!p)
        return 1;

    double e1[3], e2[3];
    int rc = Polarization_make_basis_angle(k_vec, angle_deg, e1, e2);
    if (rc != 0)
        return rc;

    double delta;
    rc = sense_to_delta(sense, &delta);
    if (rc != 0)
        return 6;

    return Polarization_init(p, e1, e2, 1.0, 1.0, delta);
}
