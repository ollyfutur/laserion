#pragma once
#include <mpi.h>
#include <stddef.h>

#include "diag_grid.h"   /* DG_Request1D/2D, DG_RunOptions */
#include "laser.h"       /* LaserPulse */
#include "inputdeck.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------- options -------------------------- */
/* Keep this struct compatible with your existing ionization_diag.c usage. */
typedef struct IonGridOptions
{
    /* Species name (e.g., "He"). Must match ionization_model.c tables. */
    const char *species;

    /* List of charge states Z for which probabilities are written, length nZ. */
    const int  *Z_list;
    size_t      nZ;

    /* Ionization model selector passed through to ION_probs_at_r(). */
    const IonizationModel *ion_model;

    /* Time integration window (fs). */
    double      tmin_fs;
    double      tmax_fs;
    double      dt_fs;

    /* If nonzero, also write total ionization probability. */
    int         write_total;

    /* Run options for diag-grid evaluation/writing. */
    DG_RunOptions run;
} IonGridOptions;

/* Low-level entry points: you already have these. */
int iongrid_run_1d(const LaserPulse *pulse,
                   const DG_Request1D *req,
                   const IonGridOptions *ion,
                   const char *path_prefix,
                   MPI_Comm comm);

int iongrid_run_2d(const LaserPulse *pulse,
                   const DG_Request2D *req,
                   const IonGridOptions *ion,
                   const char *path_prefix,
                   MPI_Comm comm);

/* High-level integration with inputdeck/run flow:
 * - Uses sim->ionization_frac requests
 * - Uses sim->grid to define axis ranges/resolution
 * - Uses sim->run.gas + sim->run.ionization_model
 * - Writes outputs into out_dir (e.g., "MS/ioniz_frac")
 */

int iongrid_run_from_inputdeck(const InputSimSpec *sim,
                               const LaserPulse *pulse,
                               const char *out_dir,
                               MPI_Comm comm);


int iongrid_run_full_from_cache(const InputSimSpec *sim,
                                const char *cache_dir,   /* "MS/cache" */
                                const char *out_dir,     /* "MS/ioniz_frac" */
                                MPI_Comm comm);



#ifdef __cplusplus
} /* extern "C" */
#endif
