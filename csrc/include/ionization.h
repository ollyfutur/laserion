#pragma once
#include <stddef.h>

#include "laser.h"              /* LaserPulse */
#include "ionization_model.h"   /* IonizationModel */

/* Compute ionization time-series quantities for several charge states.
 *
 * This is the shared backend used by MDF (which stores w/S/dP arrays) and can
 * also be reused by other diagnostics if needed.
 *
 * Storage layout for w/S/dP: contiguous blocks [nZ][N] in row-major order:
 *   row(iz) starts at ptr + iz*N.
 *
 * Convention for dt_local: dt_local[i] = t[i+1]-t[i] for i=0..N-2 and
 * dt_local[N-1] = dt_local[N-2] (to match the Python reference behavior).
 *
 * Returns 0 on success, non-zero on error.
 */
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
                           double *P_total);

/* Compute ionization probabilities at a single spatial point r (um),
 * integrating ADK rates in time, using ONLY |E| (no A/p).
 *
 * P_levels_out must be length nZ.
 * If P_total_out != NULL, it will be filled with sum(P_levels_out).
 *
 * If ion_model == NULL, an ADKModel is constructed internally.
 *
 * Returns 0 on success, non-zero on error.
 */
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
                   double *P_total_out);

