#include "ionization.h"

#include <stdlib.h>
#include <math.h>

static void cumulative_trapezoid_1d(const double *y, const double *t, size_t N, double *out_int)
{
    out_int[0] = 0.0;
    for (size_t i = 1; i < N; ++i)
    {
        const double dt = t[i] - t[i - 1];
        out_int[i] = out_int[i - 1] + 0.5 * (y[i - 1] + y[i]) * dt;
    }
}

int ION_compute_timeseries(const IonizationModel *model,
                           const char *species,
                           const int *Z_list,
                           size_t nZ,
                           const double *t_fs,
                           size_t N,
                           const double *E_abs,
                           double *w,
                           double *S,
                           double *dP,
                           double *P_levels,
                           double *P_total)
{
    if (!model || !species || !Z_list || nZ == 0) return 1;
    if (!t_fs || !E_abs || N < 2) return 2;
    if (!w || !S || !dP || !P_levels) return 3;

    double *dt_local = (double *)malloc(N * sizeof(double));
    double *Wint     = (double *)malloc(N * sizeof(double));
    if (!dt_local || !Wint)
    {
        free(dt_local);
        free(Wint);
        return 4;
    }

    for (size_t i = 0; i + 1 < N; ++i)
        dt_local[i] = t_fs[i + 1] - t_fs[i];
    dt_local[N - 1] = dt_local[N - 2];

    double Ptot = 0.0;

    for (size_t iz = 0; iz < nZ; ++iz)
    {
        const int Z = Z_list[iz];

        double *w_row  = &w[iz * N];
        double *S_row  = &S[iz * N];
        double *dP_row = &dP[iz * N];

        /* w(t) */
        for (size_t i = 0; i < N; ++i)
        {
            double wi = IonizationModel_rate((IonizationModel *)model, E_abs[i], species, Z);
            if (!(wi >= 0.0) || !isfinite(wi)) wi = 0.0;
            w_row[i] = wi;
        }

        /* Wint(t) = ∫ w dt */
        cumulative_trapezoid_1d(w_row, t_fs, N, Wint);

        /* S(t) and dP */
        double P = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            const double Si  = exp(-Wint[i]);
            const double dpi = w_row[i] * Si * dt_local[i];
            S_row[i]  = Si;
            dP_row[i] = dpi;
            P += dpi;
        }

        P_levels[iz] = P;
        Ptot += P;
    }

    if (P_total) *P_total = Ptot;

    free(Wint);
    free(dt_local);
    return 0;
}

int ION_probs_at_r(const LaserPulse *pulse,
                   const char *species,
                   const int *Z_list,
                   size_t nZ,
                   const double r_um[3],
                   const IonizationModel *ion_model,
                   double tmin_fs,
                   double tmax_fs,
                   double dt_fs,
                   double *P_levels_out,
                   double *P_total_out)
{
    if (!pulse || !species || !Z_list || nZ == 0 || !r_um || !P_levels_out) return 1;
    if (!(dt_fs > 0.0)) return 2;
    if (!(tmax_fs > tmin_fs)) return 3;

    /* If model not provided, create a local ADK model. */
    ADKModel local_adk;
    const IonizationModel *model = ion_model;
    if (!model)
    {
        ADKModel_init(&local_adk);
        model = (const IonizationModel *)&local_adk;
    }

    /* Time grid */
    const size_t N = (size_t)ceil((tmax_fs - tmin_fs) / dt_fs) + 1;
    if (N < 2) return 4;

    double *t     = (double *)malloc(N * sizeof(double));
    double *E_abs = (double *)malloc(N * sizeof(double));
    if (!t || !E_abs)
    {
        free(t);
        free(E_abs);
        return 5;
    }

    for (size_t i = 0; i < N; ++i)
        t[i] = tmin_fs + (double)i * dt_fs;

    /* |E(t,r)| */
    for (size_t i = 0; i < N; ++i)
    {
        double E[3] = {0.0, 0.0, 0.0};
        LaserPulse_E(pulse, t[i], r_um, E);
        E_abs[i] = sqrt(E[0] * E[0] + E[1] * E[1] + E[2] * E[2]);
    }

    /* Minimal buffers: reuse the same backend used by MDF. */
    double *w  = (double *)malloc(nZ * N * sizeof(double));
    double *S  = (double *)malloc(nZ * N * sizeof(double));
    double *dP = (double *)malloc(nZ * N * sizeof(double));
    if (!w || !S || !dP)
    {
        free(t);
        free(E_abs);
        free(w);
        free(S);
        free(dP);
        return 6;
    }

    const int rc = ION_compute_timeseries(model, species, Z_list, nZ, t, N, E_abs,
                                         w, S, dP, P_levels_out, P_total_out);

    free(w);
    free(S);
    free(dP);
    free(t);
    free(E_abs);

    return rc;
}

