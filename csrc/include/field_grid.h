#pragma once
#include <stddef.h>
#include <stdint.h>
#include <mpi.h>

#include "laser.h"   /* LaserPulse_E, LaserPulse_A */
#include "diag_h5.h" /* grid writer types (DiagAxis, etc.) */

/* ----------------------------- axis selection ----------------------------- */

typedef enum FG_Axis
{
    FG_T = 1,
    FG_X = 2,
    FG_Y = 3,
    FG_Z = 4
} FG_Axis;

typedef enum FG_AxisKind
{
    FG_AXIS_LINSPACE = 1, /* min..max with n points */
    FG_AXIS_VALUES = 2    /* explicit values[] */
} FG_AxisKind;

typedef struct FG_AxisSpec
{
    FG_Axis id;
    FG_AxisKind kind;

    /* name/units are written to the H5 axis attributes */
    const char *name;  /* e.g. "t", "x", "y", "z" */
    const char *units; /* "fs" for t, "um" for x/y/z */

    /* linspace */
    double min;
    double max;
    size_t n;

    /* explicit */
    const double *values;
    size_t n_values;
} FG_AxisSpec;

typedef enum FG_QuantityMask
{
    FG_Q_E = 1u << 0,
    FG_Q_A = 1u << 1
} FG_QuantityMask;

typedef enum FG_ComponentMask
{
    FG_C_X = 1u << 0,
    FG_C_Y = 1u << 1,
    FG_C_Z = 1u << 2,
    FG_C_ALL = (FG_C_X | FG_C_Y | FG_C_Z)
} FG_ComponentMask;

typedef enum FG_WriteMode
{
    FG_WRITE_SINGLE_FILE = 1, /* one H5 file containing all datasets */
    FG_WRITE_SPLIT_FILES = 2  /* one H5 file per dataset: <prefix>_<name>.h5 */
} FG_WriteMode;

typedef struct FG_Options
{
    uint32_t quantity_mask;  /* FG_Q_E and/or FG_Q_A */
    uint32_t component_mask; /* FG_C_* */
    FG_WriteMode write_mode;
    int root_rank; /* usually 0 */
} FG_Options;

static inline FG_Options fg_default_options(void)
{
    FG_Options o;
    o.quantity_mask = (uint32_t)FG_Q_E;
    o.component_mask = (uint32_t)FG_C_ALL;
    o.write_mode = FG_WRITE_SPLIT_FILES;
    o.root_rank = 0;
    return o;
}

/* ------------------------------ 1D / 2D req ------------------------------ */

typedef struct FG_Request1D
{
    FG_AxisSpec a1;
    /* fixed coordinates for the other dimensions */
    double t0_fs;
    double x0_um;
    double y0_um;
    double z0_um;
} FG_Request1D;

typedef struct FG_Request2D
{
    FG_AxisSpec a1; /* fastest axis */
    FG_AxisSpec a2; /* slow axis */
    /* fixed coordinates for the other dimensions */
    double t0_fs;
    double x0_um;
    double y0_um;
    double z0_um;
} FG_Request2D;

/* ------------------------------ public API -------------------------------- */

/* Validate axis spec and option masks; returns 0 on success. */
int fg_validate_axis(const FG_AxisSpec *a);
int fg_validate_options(const FG_Options *opt);
int fg_validate_request_1d(const FG_Request1D *r, const FG_Options *opt);
int fg_validate_request_2d(const FG_Request2D *r, const FG_Options *opt);

/*
 * Compute and write Ex/Ey/Ez and/or Ax/Ay/Az on a 1D or 2D grid.
 *
 * - path_or_prefix:
 *   - FG_WRITE_SINGLE_FILE: this is the final file path (e.g. "out.h5")
 *   - FG_WRITE_SPLIT_FILES: this is a prefix (e.g. "out/grid") -> "out/grid_Ex.h5" etc
 *
 * - time/iter in H5 output are set to 0 (as requested).
 * - axis units: "fs" for t; "um" for x/y/z.
 * - E units: "GV/m"; A units: "a.u." (placeholder string; keep consistent until you standardize).
 */
int fg_run_1d(const struct LaserPulse *pulse,
              const FG_Request1D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm);

int fg_run_2d(const struct LaserPulse *pulse,
              const FG_Request2D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm);
