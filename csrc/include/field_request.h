/* ============================================================================
 * File: field_request.h
 *
 * Purpose:
 *   Define "what to compute" (FieldRequest) independently of:
 *     - how it is computed (field_calc_*.c)
 *     - how it is written (diag_h5.c)
 *
 * Notes:
 *   - Units: keep consistent with your codebase conventions:
 *       t in fs, r in um.
 *   - Data type choices: double for input grids/coordinates.
 *   - Components are requested via a bitmask.
 * ============================================================================
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------- Core enums -------------------------------- */

typedef enum FieldQuantity
{
    FIELD_Q_E = 1,   /* Electric field E */
    FIELD_Q_A = 2    /* Vector potential A */
} FieldQuantity;

/* Bitmask for selecting vector components */
typedef enum FieldComponents
{
    FIELD_C_X = 1u << 0,
    FIELD_C_Y = 1u << 1,
    FIELD_C_Z = 1u << 2,
    FIELD_C_ALL = FIELD_C_X | FIELD_C_Y | FIELD_C_Z
} FieldComponents;

/* 1D / 2D domain axis identifiers (for clarity in requests) */
typedef enum FieldAxisId
{
    FIELD_AXIS_T = 0, /* time */
    FIELD_AXIS_X = 1,
    FIELD_AXIS_Y = 2,
    FIELD_AXIS_Z = 3,
    FIELD_AXIS_P = 4  /* point index (for point-lists) */
} FieldAxisId;

/* Request kind: which calculator should handle it */
typedef enum FieldRequestKind
{
    FIELD_REQ_TIMESERIES_POINTS = 1, /* points × time (2D) */
    FIELD_REQ_1D_LINEOUT        = 2, /* 1D axis (x/y/z) at fixed others, fixed t */
    FIELD_REQ_2D_SLICE          = 3  /* 2D plane (xy/xz/yz) at fixed coord, fixed t */
} FieldRequestKind;

/* How calculators should treat MPI output staging.
 * (Writer policy is still outside, but calculators need to know whether they must
 * assemble a full global array on rank 0.)
 */
typedef enum FieldAssemblePolicy
{
    FIELD_ASSEMBLE_GATHER_TO_ROOT = 1, /* compute local, gather global to root */
    FIELD_ASSEMBLE_KEEP_DISTRIBUTED = 2 /* compute local only (future: parallel HDF5) */
} FieldAssemblePolicy;

/* ------------------------------ Grid defs -------------------------------- */

typedef struct FieldLinspace
{
    double min;      /* inclusive */
    double max;      /* inclusive (interpretation depends on generator) */
    size_t n;        /* number of points, must be >= 2 */
} FieldLinspace;

/* A concrete axis either comes from explicit values, or from a linspace. */
typedef enum FieldAxisSpecKind
{
    FIELD_AXIS_VALUES  = 1,
    FIELD_AXIS_LINSPACE = 2
} FieldAxisSpecKind;

typedef struct FieldAxisSpec
{
    FieldAxisId id;               /* semantic label: x, y, z, t, p */
    const char *name;             /* e.g. "x", "t" (optional, may be NULL) */
    const char *units;            /* e.g. "um", "fs" (optional, may be NULL) */

    FieldAxisSpecKind kind;

    /* If kind == FIELD_AXIS_VALUES */
    const double *values;         /* length = n */
    size_t n;

    /* If kind == FIELD_AXIS_LINSPACE */
    FieldLinspace lin;
} FieldAxisSpec;

/* --------------------------- Request payloads ---------------------------- */

/* (1) time-series at a list of points: axes are (p, t) */
typedef struct FieldRequestTimeseriesPoints
{
    /* Points in um: array [np][3] */
    const double (*points_um)[3];
    size_t np;

    /* Time axis */
    FieldAxisSpec t; /* id must be FIELD_AXIS_T */
} FieldRequestTimeseriesPoints;

/* (2) 1D lineout: one varying axis among x/y/z, fixed other coords, fixed time */
typedef struct FieldRequest1DLineout
{
    FieldAxisId axis;  /* FIELD_AXIS_X / Y / Z */

    /* varying coordinate axis values */
    FieldAxisSpec a1;  /* id must match 'axis' */

    /* fixed coordinates (um) for the two non-varying spatial axes */
    double x_um;
    double y_um;
    double z_um;

    /* fixed time (fs) */
    double t_fs;
} FieldRequest1DLineout;

/* (3) 2D slice: varying plane among (x,y), (x,z), (y,z); fixed remaining coord; fixed time */
typedef enum FieldPlane2D
{
    FIELD_PLANE_XY = 1,
    FIELD_PLANE_XZ = 2,
    FIELD_PLANE_YZ = 3
} FieldPlane2D;

typedef struct FieldRequest2DSlice
{
    FieldPlane2D plane;

    /* axis specs for the two varying coordinates */
    FieldAxisSpec a1; /* id must match plane first axis */
    FieldAxisSpec a2; /* id must match plane second axis */

    /* fixed coordinate (um) for the orthogonal axis */
    double fixed_um;

    /* fixed time (fs) */
    double t_fs;
} FieldRequest2DSlice;

/* ------------------------------- Top-level ------------------------------- */

/* Global request options shared by all kinds */
typedef struct FieldRequestOptions
{
    /* What to compute */
    uint32_t quantity_mask;      /* bitwise OR of FieldQuantity values (E/A) */
    uint32_t component_mask;     /* bitwise OR of FieldComponents values (X/Y/Z) */

    /* MPI assembly preference */
    FieldAssemblePolicy assemble;

    /* Root rank for gather-to-root */
    int root_rank;

    /* Hints / metadata (optional; writers may store them as attributes) */
    const char *label;           /* e.g. "probe_01", "xy_focus" */
} FieldRequestOptions;

typedef struct FieldRequest
{
    FieldRequestKind kind;
    FieldRequestOptions opt;

    union
    {
        FieldRequestTimeseriesPoints ts_points;
        FieldRequest1DLineout        line1d;
        FieldRequest2DSlice          slice2d;
    } u;
} FieldRequest;

/* --------------------------- Validation helpers -------------------------- */

/* Basic validation of a request for internal consistency.
 * Returns 0 if OK; nonzero otherwise.
 * This should be called by field_calc_* entry points.
 */
int field_request_validate(const FieldRequest *req);

/* Utility: fill default options.
 * - quantity_mask = FIELD_Q_E (E only)
 * - component_mask = FIELD_C_ALL
 * - assemble = FIELD_ASSEMBLE_GATHER_TO_ROOT
 * - root_rank = 0
 * - label = NULL
 */
FieldRequestOptions field_request_default_options(void);

#ifdef __cplusplus
}
#endif

