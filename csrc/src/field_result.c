/* ============================================================================
 * File: field_result.c
 * ============================================================================
 */
#include "field_result.h"

#include <stdlib.h>
#include <string.h>

static void axis_zero(FieldAxis *a)
{
    a->id = (FieldAxisId)0;
    a->name = NULL;
    a->units = NULL;
    a->n = 0;
    a->values = NULL;
}

static void dataset_zero(FieldDataset *d)
{
    d->name = NULL;
    d->long_name = NULL;
    d->units = NULL;
    d->ndims = 0;
    d->dims[0] = d->dims[1] = 0;
    d->data = NULL;
}

void field_result_init(FieldResult *res)
{
    if (!res) return;
    res->layout = (FieldResultLayout)0;
    res->naxes = 0;
    axis_zero(&res->axes[0]);
    axis_zero(&res->axes[1]);

    res->label = NULL;

    res->has_points = 0;
    res->points_x_um = NULL;
    res->points_y_um = NULL;
    res->points_z_um = NULL;

    res->ndatasets = 0;
    res->datasets = NULL;
}

void field_result_free(FieldResult *res)
{
    if (!res) return;

    for (int i = 0; i < 2; ++i)
    {
        free(res->axes[i].values);
        res->axes[i].values = NULL;
        res->axes[i].n = 0;
    }

    if (res->datasets)
    {
        for (size_t i = 0; i < res->ndatasets; ++i)
        {
            free(res->datasets[i].data);
            res->datasets[i].data = NULL;
        }
        free(res->datasets);
    }
    res->datasets = NULL;
    res->ndatasets = 0;

    free(res->points_x_um);
    free(res->points_y_um);
    free(res->points_z_um);
    res->points_x_um = res->points_y_um = res->points_z_um = NULL;
    res->has_points = 0;

    res->label = NULL;
    res->layout = (FieldResultLayout)0;
    res->naxes = 0;
}

int field_axis_from_spec(FieldAxis *out, const FieldAxisSpec *spec)
{
    if (!out || !spec) return 1;

    out->id = spec->id;
    out->name = spec->name;
    out->units = spec->units;

    if (spec->kind == FIELD_AXIS_VALUES)
    {
        if (!spec->values || spec->n < 2) return 2;
        out->n = spec->n;
        out->values = (double*)malloc(sizeof(double) * out->n);
        if (!out->values) return 3;
        memcpy(out->values, spec->values, sizeof(double) * out->n);
        return 0;
    }
    else if (spec->kind == FIELD_AXIS_LINSPACE)
    {
        if (spec->lin.n < 2) return 4;

        out->n = spec->lin.n;
        out->values = (double*)malloc(sizeof(double) * out->n);
        if (!out->values) return 5;

        const double a = spec->lin.min;
        const double b = spec->lin.max;

        if (out->n == 1)
        {
            out->values[0] = a;
            return 0;
        }

        const double denom = (double)(out->n - 1);
        for (size_t i = 0; i < out->n; ++i)
        {
            const double s = (double)i / denom;
            out->values[i] = a + (b - a) * s;
        }
        return 0;
    }

    return 6;
}

int field_axis_make_index(FieldAxis *out, FieldAxisId id, size_t n, const char *name, const char *units)
{
    if (!out) return 1;
    if (n < 1) return 2;

    out->id = id;
    out->name = name;
    out->units = units;
    out->n = n;
    out->values = (double*)malloc(sizeof(double) * n);
    if (!out->values) return 3;

    for (size_t i = 0; i < n; ++i)
        out->values[i] = (double)i;

    return 0;
}

static size_t prod_dims(int ndims, const size_t dims[2])
{
    if (ndims == 1) return dims[0];
    if (ndims == 2) return dims[0] * dims[1];
    return 0;
}

size_t field_dataset_nelems(const FieldDataset *ds)
{
    if (!ds) return 0;
    return prod_dims(ds->ndims, ds->dims);
}

int field_result_alloc_datasets(
    FieldResult *res,
    size_t ndatasets,
    const char *const *dataset_names,
    const char *const *dataset_units,
    const char *const *long_names)
{
    if (!res) return 1;
    if (ndatasets == 0) return 2;
    if (!dataset_names || !dataset_units) return 3;

    if (!(res->naxes == 1 || res->naxes == 2)) return 4;
    if (res->axes[0].n < 1) return 5;
    if (res->naxes == 2 && res->axes[1].n < 1) return 6;

    res->datasets = (FieldDataset*)calloc(ndatasets, sizeof(FieldDataset));
    if (!res->datasets) return 7;

    res->ndatasets = ndatasets;

    for (size_t i = 0; i < ndatasets; ++i)
    {
        FieldDataset *ds = &res->datasets[i];
        dataset_zero(ds);

        ds->name = dataset_names[i];
        ds->units = dataset_units[i];
        ds->long_name = long_names ? long_names[i] : NULL;

        ds->ndims = res->naxes;
        ds->dims[0] = res->axes[0].n;
        ds->dims[1] = (res->naxes == 2) ? res->axes[1].n : 0;

        const size_t n = field_dataset_nelems(ds);
        ds->data = (float*)malloc(sizeof(float) * n);
        if (!ds->data)
        {
            /* cleanup everything allocated so far */
            for (size_t j = 0; j < i; ++j) free(res->datasets[j].data);
            free(res->datasets);
            res->datasets = NULL;
            res->ndatasets = 0;
            return 8;
        }
    }

    return 0;
}

FieldDataset *field_result_find_dataset(FieldResult *res, const char *name)
{
    if (!res || !name) return NULL;
    for (size_t i = 0; i < res->ndatasets; ++i)
    {
        if (res->datasets[i].name && strcmp(res->datasets[i].name, name) == 0)
            return &res->datasets[i];
    }
    return NULL;
}

