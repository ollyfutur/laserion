#include <stddef.h>
#include <math.h>
#include "transverse_profile.h"

/* Physicists' Hermite polynomials H_n(x), matching numpy.polynomial.hermite */
static double hermite_phys(int n, double x)
{
    if (n == 0) return 1.0;
    if (n == 1) return 2.0 * x;

    double Hnm1 = 1.0;        /* H_0 */
    double Hn   = 2.0 * x;    /* H_1 */

    for (int k = 1; k < n; ++k) {
        double Hp1 = 2.0 * x * Hn - 2.0 * k * Hnm1;
        Hnm1 = Hn;
        Hn   = Hp1;
    }
    return Hn;
}

static inline void compute_z0_k(double w0_um, double wavelength_um,
                                double *z0_um_out, double *k_out)
{
    const double pi = 3.14159265358979323846;
    const double z0 = pi * w0_um * w0_um / wavelength_um; /* Rayleigh length [µm] */
    const double k  = 2.0 * pi / wavelength_um;           /* wave number [1/µm] */
    if (z0_um_out) *z0_um_out = z0;
    if (k_out)     *k_out     = k;
}

/* ===================== PlaneWaveProfile ===================== */

static double plane_eval(const TransverseProfile *base,
                         const double r_beam_um[3],
                         double wavelength_um)
{
    (void)base;
    (void)r_beam_um;
    (void)wavelength_um;
    return 1.0;
}

static const TransverseProfileVTable PLANE_VT = {
    .eval    = plane_eval,
    .phase   = TransverseProfile_phase_default,
    .destroy = NULL
};

void PlaneWaveProfile_init(PlaneWaveProfile *p)
{
    p->base.vt = &PLANE_VT;
}

/* ===================== HermiteTransverse ===================== */

static double hermite_eval(const TransverseProfile *base,
                           const double r_beam_um[3],
                           double wavelength_um)
{
    const HermiteTransverse *self = (const HermiteTransverse *)base;

    const double x_um = r_beam_um[0];
    const double y_um = r_beam_um[1];
    const double z_um = r_beam_um[2];

    const double z_rel = z_um - self->zf_um;

    double z0_um, k;
    compute_z0_k(self->w0_um, wavelength_um, &z0_um, &k);
    (void)k; /* not used in envelope */

    const double wz = self->w0_um * sqrt(1.0 + (z_rel * z_rel) / (z0_um * z0_um));

    const double arg_x = sqrt(2.0) * x_um / wz;
    const double arg_y = sqrt(2.0) * y_um / wz;

    const double H_l = hermite_phys(self->l, arg_x);
    const double H_m = hermite_phys(self->m, arg_y);

    const double gauss = exp(-(x_um * x_um + y_um * y_um) / (wz * wz));

    return (self->w0_um / wz) * H_l * H_m * gauss;
}

static double hermite_phase(const TransverseProfile *base,
                            const double r_beam_um[3],
                            double wavelength_um)
{
    const HermiteTransverse *self = (const HermiteTransverse *)base;

    const double x_um = r_beam_um[0];
    const double y_um = r_beam_um[1];
    const double z_um = r_beam_um[2];

    const double z_rel = z_um - self->zf_um;

    double z0_um, k;
    compute_z0_k(self->w0_um, wavelength_um, &z0_um, &k);

    /* Curvature term */
    double curvature = 0.0;
    if (fabs(z_rel) > 1e-14) {
        const double R = (z_rel * z_rel + z0_um * z0_um) / z_rel;
        curvature = -k * (x_um * x_um + y_um * y_um) / (2.0 * R);
    }

    /* Gouy phase */
    const double zeta = atan2(z_rel, z0_um);
    const double gouy = (self->l + self->m + 1) * zeta;

    return curvature + gouy;
}

static const TransverseProfileVTable HERMITE_VT = {
    .eval    = hermite_eval,
    .phase   = hermite_phase,
    .destroy = NULL
};

void HermiteTransverse_init(HermiteTransverse *h,
                            double w0_um,
                            double zf_um,
                            int l,
                            int m)
{
    h->base.vt = &HERMITE_VT;
    h->w0_um   = w0_um;
    h->zf_um   = zf_um;
    h->l       = l;
    h->m       = m;
}

/* ===================== GaussianTransverse (HG_00) ===================== */

void GaussianTransverse_init(GaussianTransverse *g,
                             double w0_um,
                             double zf_um)
{
    HermiteTransverse_init(g, w0_um, zf_um, 0, 0);
}

