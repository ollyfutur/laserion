#pragma once
#include <stddef.h>
#include <mpi.h>

#include "mdf.h"
#include "laser.h"
#include "diag_h5.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MDFGridKind
{
    MDF_GRID_F_PX    = 0,
    MDF_GRID_F_PY    = 1,
    MDF_GRID_F_PZ    = 2,
    MDF_GRID_F_PX_PY = 3,
    MDF_GRID_F_PY_PZ = 4,
    MDF_GRID_F_PX_PZ = 5
} MDFGridKind;

/* Options for integrating MDF over a spatial region and binning in momentum space. */
typedef struct MDFGridOptions
{
    /* Species/levels (same convention as ionization module). */
    const char *species;
    const int  *Z_list;
    size_t      nZ;

    /* Ionization model (optional). If NULL => MDF_build creates ADKModel internally. */
    const IonizationModel *ion_model;

    /* Time window for MDF_build (required by MDF_build currently). */
    double tmin_fs;
    double tmax_fs;

    /* If <=0, MDF_build uses lambda/c/300 default for SinglePulse. */
    double dt_fs;

    /* Passed to MDF_build but currently unused there (kept for forward compatibility). */
    double envelope_cut;

    /* Spatial region [um]. */
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;

    /* Spatial sampling resolution (centers). Total samples = nx*ny*nz. */
    int nx, ny, nz;

    /* Which projection of p to histogram. */
    MDFGridKind kind;

    /* Momentum binning [m_e c]. */
    int    nbins1;
    double p1min, p1max;

    /* For 2D kinds only. */
    int    nbins2;
    double p2min, p2max;

    /* If nonzero: normalize global histogram so that sum(f)=1. */
    int normalize_sum_to_1;

    /* Output naming. Result file is: <path_prefix> + <file_suffix>. */
    const char *dataset_name; /* e.g. "f_pxpz" */
    const char *label;        /* e.g. "f(p_x,p_z)" */
    const char *units;        /* e.g. "1" (probability density or probability mass) */
    const char *file_suffix;  /* e.g. "_mdf_pxpz.h5" */

    /* Root rank for writing. */
    int root_rank;
} MDFGridOptions;

/* Compute MDF histogram integrated over a spatial region and write one H5 file.
 *
 * The file is written using diag_h5_write_grid_1d/2d.
 * - Axis min/max are written in /AXIS/AXIS1(/AXIS2).
 * - Data is float32, size nbins1 (1D) or nbins2*nbins1 (2D, C-order [p2][p1]).
 *
 * MPI:
 * - Spatial points are distributed over ranks (block decomposition).
 * - Histogram contributions are accumulated locally then summed with MPI_Allreduce.
 */
int mdfgrid_run(const LaserPulse *pulse,
                const MDFGridOptions *opt,
                const char *path_prefix,
                MPI_Comm comm);

#ifdef __cplusplus
} /* extern "C" */
#endif

