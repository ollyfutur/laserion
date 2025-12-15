#ifndef MDF_H
#define MDF_H

#include <stddef.h>
#include "base.h"
#include "ionization_model.h"

/* A [GV/m·fs] -> p [m_e c] */
#define MDF_CONV_A_TO_P (-0.0005866792097892349)

typedef struct {
    const LaserPulse *pulse;
    const IonizationModel *ion_model;

    const char *species;
    int *Z_list;
    size_t nZ;
    int ion_model_owned;
    double r[3];

    /* time grid */
    double *t;      /* (N,) */
    size_t N;

    /* A(t,r) and p(t,r) */
    double (*A)[3]; /* (N,3) */
    double (*p)[3]; /* (N,3) */

    /* |E(t,r)| */
    double *E_abs;  /* (N,) */

    /* per level arrays: stored as contiguous blocks [nZ][N] */
    double *w;      /* (nZ*N) */
    double *S;      /* (nZ*N) */
    double *dP;     /* (nZ*N) */

    /* per-level scalars */
    double *P_ion_levels; /* (nZ,) */
    double P_ion_total;
} MDF;

/*
 * Compute MDF at fixed r for species and charge states Z_list.
 *
 * If tmin_fs or tmax_fs is NaN and the pulse is a SinglePulse whose temporal
 * profile is GaussianTemporal, an automatic window is built using envelope_cut.
 *
 * Otherwise you must provide finite tmin_fs and tmax_fs.
 *
 * dt_fs: if <=0, a default dt = (lambda/c)/300 is used if the pulse provides it;
 * otherwise dt_fs must be >0.
 */
int MDF_build(MDF *m,
              const LaserPulse *pulse,
              const char *species,
              const int *Z_list,
              size_t nZ,
              const double r[3],
              const IonizationModel *ion_model, /* if NULL => ADKModel is created internally */
              double envelope_cut,
              double tmin_fs,
              double tmax_fs,
              double dt_fs);

/* Free all internally allocated buffers. */
void MDF_destroy(MDF *m);

/* Convenience: access [level, i] for w/S/dP (row-major). */
static inline double MDF_w(const MDF *m, size_t iz, size_t i)  { return m->w [iz*m->N + i]; }
static inline double MDF_S(const MDF *m, size_t iz, size_t i)  { return m->S [iz*m->N + i]; }
static inline double MDF_dP(const MDF *m, size_t iz, size_t i) { return m->dP[iz*m->N + i]; }

#endif /* MDF_H */

