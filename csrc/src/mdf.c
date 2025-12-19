#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mdf.h"
#include "laser.h"             /* for SinglePulse layout if you use dt default */
#include "ionization_model.h" /* ADKModel */
#include "ionization.h"       /* ION_compute_timeseries */

/* ---------- internal helpers ---------- */

static int is_finite(double x) { return isfinite(x) != 0; }

static void cumulative_trapezoid_vec3(const double y[][3], const double *t, size_t N, double out_int[][3])
{
    out_int[0][0] = out_int[0][1] = out_int[0][2] = 0.0;

    for (size_t i = 1; i < N; ++i)
    {
        const double dt = t[i] - t[i - 1];
        out_int[i][0] = out_int[i - 1][0] + 0.5 * (y[i - 1][0] + y[i][0]) * dt;
        out_int[i][1] = out_int[i - 1][1] + 0.5 * (y[i - 1][1] + y[i][1]) * dt;
        out_int[i][2] = out_int[i - 1][2] + 0.5 * (y[i - 1][2] + y[i][2]) * dt;
    }
}

/* ---------- public API ---------- */

int MDF_build(MDF *m,
              const LaserPulse *pulse,
              const char *species,
              const int *Z_list,
              size_t nZ,
              const double r[3],
              const IonizationModel *ion_model_in,
              double envelope_cut,
              double tmin_fs,
              double tmax_fs,
              double dt_fs)
{
    if (!m || !pulse || !species || !Z_list || nZ == 0 || !r)
        return 1;

    memset(m, 0, sizeof(*m));

    m->pulse = pulse;
    m->species = species;
    m->r[0] = r[0];
    m->r[1] = r[1];
    m->r[2] = r[2];

    /* We do not auto-window here, so envelope_cut is currently unused. */
    (void)envelope_cut;

    /* ---------- ion model ownership ---------- */
    if (ion_model_in)
    {
        m->ion_model = ion_model_in;
        m->ion_model_owned = 0;
    }
    else
    {
        ADKModel *adk = (ADKModel *)malloc(sizeof(*adk));
        if (!adk)
        {
            MDF_destroy(m);
            return 2;
        }
        ADKModel_init(adk);
        m->ion_model = (IonizationModel *)adk;
        m->ion_model_owned = 1;
    }

    /* ---------- copy Z list ---------- */
    m->Z_list = (int *)malloc(nZ * sizeof(int));
    if (!m->Z_list)
    {
        MDF_destroy(m);
        return 2;
    }
    for (size_t i = 0; i < nZ; ++i)
        m->Z_list[i] = Z_list[i];
    m->nZ = nZ;

    /* ---------- time window: require explicit ---------- */
    double tmin = tmin_fs;
    double tmax = tmax_fs;

    if (!(is_finite(tmin) && is_finite(tmax)))
    {
        MDF_destroy(m);
        return 3; /* explicit tmin/tmax required */
    }
    if (!(tmax > tmin))
    {
        MDF_destroy(m);
        return 4;
    }

    /* ---------- dt default: period/300 if we can infer lambda and c from SinglePulse ---------- */
    double dt = dt_fs;
    if (!(dt > 0.0))
    {
        const SinglePulse *sp = (const SinglePulse *)pulse;
        const double c = (sp->c_um_per_fs > 0.0) ? sp->c_um_per_fs : 0.299792458;
        const double lambda = sp->wavelength_um;
        if (!(lambda > 0.0))
        {
            MDF_destroy(m);
            return 5;
        }
        const double T = lambda / c;
        dt = T / 300.0;
    }

    /* ---------- grid size ---------- */
    size_t N = (size_t)ceil((tmax - tmin) / dt) + 1;
    if (N < 2)
    {
        MDF_destroy(m);
        return 6;
    }

    m->N = N;

    m->t = (double *)malloc(N * sizeof(double));
    m->E_abs = (double *)malloc(N * sizeof(double));
    m->A = (double (*)[3])malloc(N * sizeof(double[3]));
    m->p = (double (*)[3])malloc(N * sizeof(double[3]));
    if (!m->t || !m->E_abs || !m->A || !m->p)
    {
        MDF_destroy(m);
        return 7;
    }

    for (size_t i = 0; i < N; ++i)
    {
        m->t[i] = tmin + (double)i * dt;
    }

    /* ---------- E grid (vector) for A integration ---------- */
    double (*Egrid_mut)[3] = malloc(N * sizeof(double[3]));
    if (!Egrid_mut)
    {
        MDF_destroy(m);
        return 8;
    }

    for (size_t i = 0; i < N; ++i)
    {
        double Ei[3];
        LaserPulse_E(pulse, m->t[i], m->r, Ei);

        Egrid_mut[i][0] = Ei[0];
        Egrid_mut[i][1] = Ei[1];
        Egrid_mut[i][2] = Ei[2];

        m->E_abs[i] = sqrt(Ei[0] * Ei[0] + Ei[1] * Ei[1] + Ei[2] * Ei[2]);
    }

    /* ---------- A(t) = -∫ E dt ---------- */
    const double (*Egrid)[3] = (const double (*)[3])Egrid_mut;
    cumulative_trapezoid_vec3(Egrid, m->t, N, m->A);

    for (size_t i = 0; i < N; ++i)
    {
        m->A[i][0] = -m->A[i][0];
        m->A[i][1] = -m->A[i][1];
        m->A[i][2] = -m->A[i][2];

        m->p[i][0] = m->A[i][0] * MDF_CONV_A_TO_P;
        m->p[i][1] = m->A[i][1] * MDF_CONV_A_TO_P;
        m->p[i][2] = m->A[i][2] * MDF_CONV_A_TO_P;
    }

    free(Egrid_mut);

    /* ---------- allocate per-level arrays ---------- */
    m->w = (double *)malloc(nZ * N * sizeof(double));
    m->S = (double *)malloc(nZ * N * sizeof(double));
    m->dP = (double *)malloc(nZ * N * sizeof(double));
    m->P_ion_levels = (double *)malloc(nZ * sizeof(double));
    if (!m->w || !m->S || !m->dP || !m->P_ion_levels)
    {
        MDF_destroy(m);
        return 9;
    }

    /* Ionization time integration (delegated to ionization.c). */
    const int ion_rc = ION_compute_timeseries(m->ion_model,
                                             species,
                                             m->Z_list, nZ,
                                             m->t, N,
                                             m->E_abs,
                                             m->w,
                                             m->S,
                                             m->dP,
                                             m->P_ion_levels,
                                             &m->P_ion_total);
    if (ion_rc != 0)
    {
        MDF_destroy(m);
        return 10;
    }

    return 0;
}

void MDF_destroy(MDF *m)
{
    if (!m)
        return;

    if (m->ion_model_owned && m->ion_model)
    {
        /* We know we allocated an ADKModel when owned. */
        free((void *)m->ion_model);
    }

    free(m->Z_list);
    free(m->t);
    free(m->E_abs);
    free(m->A);
    free(m->p);
    free(m->w);
    free(m->S);
    free(m->dP);
    free(m->P_ion_levels);

    memset(m, 0, sizeof(*m));
}

