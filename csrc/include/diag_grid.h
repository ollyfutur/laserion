#pragma once
#include <stddef.h>
#include <mpi.h>

#include "diag_h5.h"  /* diag_h5_write_grid_1d/2d and DiagAxis/DiagFixedCoords */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------- Axis specification --------------------------- */

typedef enum DG_AxisID
{
    DG_AXIS_T = 0,
    DG_AXIS_X = 1,
    DG_AXIS_Y = 2,
    DG_AXIS_Z = 3
} DG_AxisID;

typedef enum DG_AxisKind
{
    DG_AXIS_LINSPACE = 0,
    DG_AXIS_VALUES   = 1
} DG_AxisKind;

typedef struct DG_AxisSpec
{
    DG_AxisID   id;
    DG_AxisKind kind;

    /* Optional metadata (may be NULL). */
    const char *name;   /* long name */
    const char *units;

    /* For DG_AXIS_LINSPACE */
    double min;
    double max;
    size_t n;

    /* For DG_AXIS_VALUES */
    const double *values;
    size_t        n_values;
} DG_AxisSpec;

typedef struct DG_FixedCoords
{
    double t_fs;
    double x_um;
    double y_um;
    double z_um;
} DG_FixedCoords;

/* 1D and 2D requests: at evaluation time, coords are produced by taking
 * fixed coords and overriding the relevant axis component(s). */
typedef struct DG_Request1D
{
    DG_AxisSpec     a1;
    DG_FixedCoords  fixed;
} DG_Request1D;

typedef struct DG_Request2D
{
    DG_AxisSpec     a1; /* fast axis (dimension 0) */
    DG_AxisSpec     a2; /* slow axis (dimension 1) */
    DG_FixedCoords  fixed;
} DG_Request2D;

/* -------------------------- Dataset description -------------------------- */

typedef struct DG_DatasetDesc
{
    const char *dset_name;  /* dataset name inside H5 file */
    const char *units;      /* e.g. "GV/m", "1", etc. */
    const char *label;      /* human label (your writer stores it) */

    /* Output path is: <prefix> + suffix */
    const char *file_suffix;  /* e.g. "_Ex.h5", "_Pion_Z2.h5" */
} DG_DatasetDesc;

/* ------------------------- Evaluation callback --------------------------- */

/* For each point, you produce ndatasets scalar values (float) in out_vals[].
 * coords are in fs/um using the same conventions as your codebase.
 * Return 0 for success; non-zero will store NaNs in output for that point.
 */
typedef int (*DG_EvalFn)(const DG_FixedCoords *coords, void *ctx,
                         float *out_vals, size_t ndatasets);

/* --------------------------- Run functions -------------------------------- */

/* Options for MPI and writing. */
typedef struct DG_RunOptions
{
    int root_rank;         /* usually 0 */
    int write_time_iter_0; /* if nonzero, write time=0 iter=0 (your convention) */
} DG_RunOptions;

DG_RunOptions dg_default_run_options(void);

/* Validate only (0 OK). */
int dg_validate_axis(const DG_AxisSpec *a);
int dg_validate_req_1d(const DG_Request1D *req);
int dg_validate_req_2d(const DG_Request2D *req);

/* Main entry points: evaluate and write ndatasets outputs on the grid. */
int dg_run_1d(const DG_Request1D *req,
              const DG_DatasetDesc *datasets,
              size_t ndatasets,
              DG_EvalFn eval_fn,
              void *eval_ctx,
              const DG_RunOptions *opt,
              const char *path_prefix,
              MPI_Comm comm);

int dg_run_2d(const DG_Request2D *req,
              const DG_DatasetDesc *datasets,
              size_t ndatasets,
              DG_EvalFn eval_fn,
              void *eval_ctx,
              const DG_RunOptions *opt,
              const char *path_prefix,
              MPI_Comm comm);

#ifdef __cplusplus
} /* extern "C" */
#endif

