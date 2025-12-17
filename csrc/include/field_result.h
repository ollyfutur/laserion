/* ============================================================================
 * File: field_result.h
 *
 * Purpose:
 *   Standard in-memory container for computed field data.
 *   Produced by field_calc_* modules; consumed by diag writers (diag_h5, etc.).
 *
 * Conventions:
 *   - Axes are stored as explicit value arrays (double).
 *   - Datasets are stored as contiguous float arrays with row-major ordering:
 *       For 1D: [n1]
 *       For 2D: [n2][n1]  (axis1 is fastest)
 *       For 2D timeseries (p,t): [np][nt] (t is fastest)
 *   - Multiple datasets (Ex, Ey, ...) share the same axes and shape.
 *
 * Units:
 *   - Axes values follow request conventions (t in fs; x/y/z in um).
 *   - Dataset units are stored as strings; writers may also store as attributes.
 * ============================================================================
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "field_request.h"

/* ----------------------------- Axis object -------------------------------- */

typedef struct FieldAxis
{
    FieldAxisId id;
    const char *name;   /* e.g. "x" */
    const char *units;  /* e.g. "um" */

    size_t n;
    double *values;     /* length n, owned by FieldResult */
} FieldAxis;

/* --------------------------- Dataset object ------------------------------- */

typedef struct FieldDataset
{
    /* e.g. "fields/Ex", "fields/Ay", or short "Ex" (writer can remap) */
    const char *name;

    /* human-readable long name, optional */
    const char *long_name;

    /* units, e.g. "GV/m" for E, "a.u." for A */
    const char *units;

    /* shape: ndims in {1,2}; dims follow axes order in FieldResult */
    int ndims;
    size_t dims[2];     /* dims[0]=axis1.n, dims[1]=axis2.n if ndims==2 */

    /* data pointer, owned by FieldResult */
    float *data;
} FieldDataset;

/* ---------------------------- Result object ------------------------------- */

typedef enum FieldResultLayout
{
    FIELD_RES_1D = 1,        /* one axis */
    FIELD_RES_2D = 2         /* two axes */
} FieldResultLayout;

typedef struct FieldResult
{
    FieldResultLayout layout;

    /* axes[0] is axis1 (fastest varying in memory), axes[1] axis2 if present */
    FieldAxis axes[2];
    int naxes;               /* 1 or 2 */

    /* Optional metadata */
    const char *label;       /* copied from request (non-owning by default) */

    /* Point coordinates for timeseries-points (optional).
     * If present, length must equal axes[1] when axis2 is FIELD_AXIS_P.
     * Stored as float for compactness; writer can store as 1D datasets.
     */
    int has_points;
    float *points_x_um;      /* length np */
    float *points_y_um;      /* length np */
    float *points_z_um;      /* length np */

    /* Datasets */
    size_t ndatasets;
    FieldDataset *datasets;  /* array length ndatasets, owned */
} FieldResult;

/* -------------------------- Lifecycle helpers ----------------------------- */

/* Initialize an empty FieldResult (zeroed pointers). */
void field_result_init(FieldResult *res);

/* Free all owned memory inside FieldResult (axes values, datasets, points). */
void field_result_free(FieldResult *res);

/* ---------------------- Axis materialization helpers ---------------------- */

/* Materialize an axis spec into an owned FieldAxis inside 'out'.
 * Allocates out->values and fills out->n. Copies id/name/units pointers.
 * Returns 0 on success, nonzero on error.
 */
int field_axis_from_spec(FieldAxis *out, const FieldAxisSpec *spec);

/* Convenience to create a simple integer axis 0..n-1 (double) for point index.
 * Name/units may be NULL; writer can override.
 */
int field_axis_make_index(FieldAxis *out, FieldAxisId id, size_t n, const char *name, const char *units);

/* ------------------------ Dataset allocation helpers ---------------------- */

/* Allocate datasets array and individual dataset data buffers.
 *
 * - layout: FIELD_RES_1D or FIELD_RES_2D
 * - naxes must match layout (1 or 2)
 * - axes must already be materialized (axes[i].n set)
 *
 * dataset_names: array of C strings (non-owning)
 * dataset_units: array of C strings (non-owning)
 * long_names:    optional array (may be NULL)
 *
 * Returns 0 on success.
 */
int field_result_alloc_datasets(
    FieldResult *res,
    size_t ndatasets,
    const char *const *dataset_names,
    const char *const *dataset_units,
    const char *const *long_names);

/* Find dataset by name (exact match). Returns NULL if not found. */
FieldDataset *field_result_find_dataset(FieldResult *res, const char *name);

/* ----------------------------- Utility ----------------------------------- */

/* Total number of elements in one dataset (product of dims). */
size_t field_dataset_nelems(const FieldDataset *ds);

#ifdef __cplusplus
}
#endif

