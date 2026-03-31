#include <math.h>
#include <string.h>
#include "laser.h"

/* ---------------- vector helpers ---------------- */

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

/* ---------------- SinglePulse internals ---------------- */

static void singlepulse_axis_coords(const SinglePulse *p,
                                    const double r_lab[3],
                                    double dr[3],
                                    double *z_prime_out)
{
    dr[0] = r_lab[0] - p->r_start[0];
    dr[1] = r_lab[1] - p->r_start[1];
    dr[2] = r_lab[2] - p->r_start[2];
    *z_prime_out = vdot(dr, p->k_hat);
}

static void singlepulse_to_beam_frame(const SinglePulse *p,
                                      const double r_lab[3],
                                      double r_beam[3])
{
    double dr[3], z_prime;
    singlepulse_axis_coords(p, r_lab, dr, &z_prime);

    r_beam[0] = vdot(dr, p->e_xb); /* x' */
    r_beam[1] = vdot(dr, p->e_yb); /* y' */
    r_beam[2] = z_prime;           /* z' */
}

static double singlepulse_t_eff(const SinglePulse *p,
                                double t_fs,
                                const double r_lab[3])
{
    if (!p->use_retarded_time)
        return t_fs;

    double dr[3], z_prime;
    singlepulse_axis_coords(p, r_lab, dr, &z_prime);
    return t_fs - z_prime / p->c_um_per_fs;
}

/* ---------------- LaserPulse vtable: E and A ---------------- */

/*
 * Core E evaluation with an extra phase offset added to the carrier.
 * Used directly for E (extra_phase=0) and for the quadrature field (extra_phase=π/2).
 */
static void singlepulse_E_core(const SinglePulse *p,
                               double t_fs,
                               const double r_lab[3],
                               double extra_phase,
                               double out_E[3])
{
    double r_beam[3];
    singlepulse_to_beam_frame(p, r_lab, r_beam);

    double t_eff = singlepulse_t_eff(p, t_fs, r_lab);

    double env_t = TemporalProfile_eval(p->temporal, t_eff);
    double env_r = TransverseProfile_eval(p->transverse, r_beam, p->wavelength_um);

    double phi_profile = TransverseProfile_phase(p->transverse, r_beam, p->wavelength_um);
    double phase = p->omega_rad_per_fs * t_eff + (phi_profile + p->phase0) + extra_phase;

    double pol_vec[3];
    Polarization_vector(&p->polarization, phase, pol_vec);

    double scale = p->E0 * env_t * env_r;

    out_E[0] = scale * pol_vec[0];
    out_E[1] = scale * pol_vec[1];
    out_E[2] = scale * pol_vec[2];
}

static void singlepulse_E_impl(const LaserPulse *base,
                               double t_fs,
                               const double r_lab[3],
                               double out_E[3])
{
    const SinglePulse *p = (const SinglePulse *)base;

    singlepulse_E_core(p, t_fs, r_lab, 0.0, out_E);

    if (!p->use_maxwell_correction)
        return;

    /*
     * Paraxial Maxwell correction: add longitudinal component E_∥ along k_hat
     * to satisfy ∇·E = 0 in vacuum.
     *
     *   E_∥ = -(1/k) (∂E_qx'/∂x' + ∂E_qy'/∂y')
     *
     * where E_q = E evaluated with carrier phase shifted by π/2 (quadrature
     * field — the Hilbert transform of E in the slowly-varying-envelope sense),
     * x' and y' are the beam-frame transverse coordinates (e_xb, e_yb),
     * and k = 2π/λ.  The transverse divergence is computed via central
     * finite differences with step h in the beam-frame transverse plane.
     *
     * Note: shifting along e_xb or e_yb does not change z' = (r-r_start)·k_hat
     * (since e_xb ⊥ k_hat), so the retarded time is unaffected by the shift —
     * only the transverse profile contributes to the gradient, as expected.
     */
    const double pi = 3.14159265358979323846;
    const double k  = 2.0 * pi / p->wavelength_um;
    const double h  = p->maxwell_fd_h_um > 0.0
                          ? p->maxwell_fd_h_um
                          : p->wavelength_um * 0.01;

    /* Offsets along beam-frame transverse axes */
    double r_px[3], r_mx[3], r_py[3], r_my[3];
    for (int j = 0; j < 3; ++j)
    {
        r_px[j] = r_lab[j] + h * p->e_xb[j];
        r_mx[j] = r_lab[j] - h * p->e_xb[j];
        r_py[j] = r_lab[j] + h * p->e_yb[j];
        r_my[j] = r_lab[j] - h * p->e_yb[j];
    }

    /* Quadrature field (phase + π/2) at the four offset points */
    double Eq_px[3], Eq_mx[3], Eq_py[3], Eq_my[3];
    singlepulse_E_core(p, t_fs, r_px, pi * 0.5, Eq_px);
    singlepulse_E_core(p, t_fs, r_mx, pi * 0.5, Eq_mx);
    singlepulse_E_core(p, t_fs, r_py, pi * 0.5, Eq_py);
    singlepulse_E_core(p, t_fs, r_my, pi * 0.5, Eq_my);

    /* Central differences: ∂E_qx'/∂x' and ∂E_qy'/∂y' */
    double dEqx_dx = (vdot(Eq_px, p->e_xb) - vdot(Eq_mx, p->e_xb)) / (2.0 * h);
    double dEqy_dy = (vdot(Eq_py, p->e_yb) - vdot(Eq_my, p->e_yb)) / (2.0 * h);

    double E_long = -(1.0 / k) * (dEqx_dx + dEqy_dy);

    out_E[0] += E_long * p->k_hat[0];
    out_E[1] += E_long * p->k_hat[1];
    out_E[2] += E_long * p->k_hat[2];
}

/*
 * Minimal A(t,r): A(t) = -∫ E(t',r) dt' from tmin to t (or to tmax).
 * Uses trapezoid on uniform grid. Deterministic, no caching/interp yet.
 *
 * If A not enabled, returns zero vector.
 */
static void singlepulse_A_impl(const LaserPulse *base,
                               double t_fs,
                               const double r_lab[3],
                               double out_A[3])
{
    const SinglePulse *p = (const SinglePulse *)base;

    out_A[0] = out_A[1] = out_A[2] = 0.0;
    if (!p->A_enabled)
        return;

    double tmin = p->A_tmin_fs;
    double tmax = p->A_tmax_fs;
    double dt = p->A_dt_fs;

    if (dt <= 0.0 || tmax <= tmin)
        return;

    /* clamp integration upper limit */
    double t_end = t_fs;
    if (t_end < tmin)
        t_end = tmin;
    if (t_end > tmax)
        t_end = tmax;

    /* integrate from tmin to t_end */
    size_t n_steps = (size_t)floor((t_end - tmin) / dt);
    double t0 = tmin;

    double E_prev[3];
    singlepulse_E_impl(base, t0, r_lab, E_prev);

    for (size_t i = 1; i <= n_steps; ++i)
    {
        double t1 = tmin + (double)i * dt;
        double E_cur[3];
        singlepulse_E_impl(base, t1, r_lab, E_cur);

        /* trapezoid: A -= 0.5*(E_prev + E_cur)*dt */
        out_A[0] -= 0.5 * (E_prev[0] + E_cur[0]) * dt;
        out_A[1] -= 0.5 * (E_prev[1] + E_cur[1]) * dt;
        out_A[2] -= 0.5 * (E_prev[2] + E_cur[2]) * dt;

        E_prev[0] = E_cur[0];
        E_prev[1] = E_cur[1];
        E_prev[2] = E_cur[2];
    }

    /* final fractional step to exactly reach t_end (if needed) */
    double t_reached = tmin + (double)n_steps * dt;
    double dt_last = t_end - t_reached;
    if (dt_last > 0.0)
    {
        double E_cur[3];
        singlepulse_E_impl(base, t_end, r_lab, E_cur);

        out_A[0] -= 0.5 * (E_prev[0] + E_cur[0]) * dt_last;
        out_A[1] -= 0.5 * (E_prev[1] + E_cur[1]) * dt_last;
        out_A[2] -= 0.5 * (E_prev[2] + E_cur[2]) * dt_last;
    }
}

static void singlepulse_destroy_impl(LaserPulse *base)
{
    (void)base; /* no heap allocations */
}

static const LaserPulseVTable SINGLEPULSE_VT = {
    .E = singlepulse_E_impl,
    .A = singlepulse_A_impl,
    .destroy = singlepulse_destroy_impl};

int SinglePulse_init(SinglePulse *p,
                     double E0,
                     double wavelength_um,
                     const TemporalProfile *temporal,
                     const TransverseProfile *transverse,
                     const Polarization *polarization,
                     double phase0,
                     const double k_vec[3],
                     int use_retarded_time,
                     const double r_start[3])
{
    if (!p || !temporal || !transverse || !polarization || !k_vec || !r_start)
        return 1;

    p->base.vt = &SINGLEPULSE_VT;

    p->E0 = E0;
    p->wavelength_um = wavelength_um;
    p->phase0 = phase0;

    if (vnormalize(k_vec, p->k_hat) != 0)
        return 2;

    if (!(wavelength_um > 0.0))
        return 5;

    p->r_start[0] = r_start[0];
    p->r_start[1] = r_start[1];
    p->r_start[2] = r_start[2];

    p->use_retarded_time = use_retarded_time ? 1 : 0;

    /* constants */
    p->c_um_per_fs = 0.299792458;
    p->omega_rad_per_fs = 2.0 * 3.14159265358979323846 * p->c_um_per_fs / wavelength_um;

    /* beam-frame transverse basis tied to lab frame (same logic as Python) */
    const double z_hat[3] = {0.0, 0.0, 1.0};
    double e1[3];

    if (fabs(vdot(z_hat, p->k_hat)) < 0.999999)
    {
        /* e1 = z_hat - (z_hat·k_hat) k_hat */
        double proj = vdot(z_hat, p->k_hat);
        e1[0] = z_hat[0] - proj * p->k_hat[0];
        e1[1] = z_hat[1] - proj * p->k_hat[1];
        e1[2] = z_hat[2] - proj * p->k_hat[2];
        if (vnormalize(e1, e1) != 0)
            return 3;
    }
    else
    {
        /* fallback to x */
        e1[0] = 1.0;
        e1[1] = 0.0;
        e1[2] = 0.0;
    }

    double e2[3];
    vcross(p->k_hat, e1, e2);
    if (vnormalize(e2, e2) != 0)
        return 4;

    p->e_xb[0] = e1[0];
    p->e_xb[1] = e1[1];
    p->e_xb[2] = e1[2];
    p->e_yb[0] = e2[0];
    p->e_yb[1] = e2[1];
    p->e_yb[2] = e2[2];

    p->temporal = temporal;
    p->transverse = transverse;
    p->polarization = *polarization; /* copy value */

    /* A disabled by default */
    p->A_enabled = 0;
    p->A_tmin_fs = 0.0;
    p->A_tmax_fs = 0.0;
    p->A_dt_fs = 0.0;

    /* Maxwell correction disabled by default */
    p->use_maxwell_correction = 0;
    p->maxwell_fd_h_um = 0.0;

    return 0;
}

void SinglePulse_enable_A(SinglePulse *p, double tmin_fs, double tmax_fs, double dt_fs)
{
    p->A_enabled = 1;
    p->A_tmin_fs = tmin_fs;
    p->A_tmax_fs = tmax_fs;
    p->A_dt_fs = dt_fs;
}

void SinglePulse_enable_maxwell_correction(SinglePulse *p, double fd_h_um)
{
    p->use_maxwell_correction = 1;
    p->maxwell_fd_h_um = fd_h_um; /* 0 = auto (wavelength/100 at eval time) */
}

/* ---------------- MultiPulse implementation ---------------- */

static void multipulse_E_impl(const LaserPulse *base,
                              double t_fs,
                              const double r_lab[3],
                              double out_E[3])
{
    const MultiPulse *m = (const MultiPulse *)base;

    out_E[0] = out_E[1] = out_E[2] = 0.0;

    for (size_t i = 0; i < m->count; ++i)
    {
        double Ei[3];
        LaserPulse_E(m->pulses[i], t_fs, r_lab, Ei);
        out_E[0] += Ei[0];
        out_E[1] += Ei[1];
        out_E[2] += Ei[2];
    }
}

/* For now MultiPulse.A uses numeric integration of E_total as well. */
static void multipulse_A_impl(const LaserPulse *base,
                              double t_fs,
                              const double r_lab[3],
                              double out_A[3])
{
    const MultiPulse *m = (const MultiPulse *)base;

    out_A[0] = out_A[1] = out_A[2] = 0.0;
    if (!m->A_enabled)
        return;

    double tmin = m->A_tmin_fs;
    double tmax = m->A_tmax_fs;
    double dt   = m->A_dt_fs;

    if (dt <= 0.0 || tmax <= tmin)
        return;

    /* clamp integration upper limit */
    double t_end = t_fs;
    if (t_end < tmin)
        t_end = tmin;
    if (t_end > tmax)
        t_end = tmax;

    /* integrate from tmin to t_end */
    size_t n_steps = (size_t)floor((t_end - tmin) / dt);
    double t0 = tmin;

    double E_prev[3];
    LaserPulse_E(base, t0, r_lab, E_prev);

    for (size_t i = 1; i <= n_steps; ++i)
    {
        double t1 = tmin + (double)i * dt;
        double E_cur[3];
        LaserPulse_E(base, t1, r_lab, E_cur);

        /* trapezoid: A -= 0.5*(E_prev + E_cur)*dt */
        out_A[0] -= 0.5 * (E_prev[0] + E_cur[0]) * dt;
        out_A[1] -= 0.5 * (E_prev[1] + E_cur[1]) * dt;
        out_A[2] -= 0.5 * (E_prev[2] + E_cur[2]) * dt;

        E_prev[0] = E_cur[0];
        E_prev[1] = E_cur[1];
        E_prev[2] = E_cur[2];
    }

    /* final fractional step to exactly reach t_end (if needed) */
    double t_reached = tmin + (double)n_steps * dt;
    double dt_last = t_end - t_reached;
    if (dt_last > 0.0)
    {
        double E_cur[3];
        LaserPulse_E(base, t_end, r_lab, E_cur);

        out_A[0] -= 0.5 * (E_prev[0] + E_cur[0]) * dt_last;
        out_A[1] -= 0.5 * (E_prev[1] + E_cur[1]) * dt_last;
        out_A[2] -= 0.5 * (E_prev[2] + E_cur[2]) * dt_last;
    }
}

static void multipulse_destroy_impl(LaserPulse *base)
{
    (void)base;
}

static const LaserPulseVTable MULTIPULSE_VT = {
    .E = multipulse_E_impl,
    .A = multipulse_A_impl,
    .destroy = multipulse_destroy_impl};

int MultiPulse_init(MultiPulse *m, const LaserPulse **pulses, size_t count)
{
    if (!m || !pulses || count == 0)
        return 1;

    m->base.vt = &MULTIPULSE_VT;
    m->pulses = pulses;
    m->count = count;

    /* A disabled by default */
    m->A_enabled = 0;
    m->A_tmin_fs = 0.0;
    m->A_tmax_fs = 0.0;
    m->A_dt_fs   = 0.0;

    return 0;
}

void MultiPulse_enable_A(MultiPulse *m, double tmin_fs, double tmax_fs, double dt_fs)
{
    m->A_enabled = 1;
    m->A_tmin_fs = tmin_fs;
    m->A_tmax_fs = tmax_fs;
    m->A_dt_fs   = dt_fs;
}
