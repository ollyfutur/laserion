#pragma once
#include <stddef.h>
#include <mpi.h>

#include "laser.h"
#include "ionization_model.h"
#include "diag_grid.h"

typedef struct IonGridOptions
{
    const char *species;
    const int  *Z_list;
    size_t      nZ;

    const IonizationModel *ion_model; /* may be NULL -> local ADK */

    double tmin_fs;
    double tmax_fs;
    double dt_fs;

    int write_total;
    DG_RunOptions run; /* root rank, time/iter policy */
} IonGridOptions;

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

