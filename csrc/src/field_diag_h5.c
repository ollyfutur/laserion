// field_diag_h5.c
#include "field_diag_h5.h"
#include "diag_h5.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int field_diag_h5_write(const char *path, const FieldResult *res)
{
    if (!path || !res)
        return 1;

    // Only implement what you need for the first test:
    // - timeseries for a single point => write 1D datasets vs t
    if (!(res->naxes == 2 || res->naxes == 1))
        return 2;

    // Expect time axis in axes[0]
    const FieldAxis *t = &res->axes[0];
    if (!t->values || t->n < 2)
        return 3;

    // Write time axis
    DiagAxis1D axis_t = {
        .name = t->name ? t->name : "t",
        .long_name = "t",
        .units = t->units ? t->units : "fs",
        .vmin = t->values[0],
        .vmax = t->values[t->n - 1]};

    // If this is a 2D result with point axis, and np==1, collapse to 1D
    size_t np = 1;
    if (res->naxes == 2)
    {
        if (res->axes[0].id != FIELD_AXIS_T)
            return 8;
        if (res->axes[1].id != FIELD_AXIS_P)
            return 9;
        if (res->axes[1].n != 1)
            return 5; // keep your current limitation
    }
    if (res->naxes == 2)
        np = res->axes[1].n;

    if (res->naxes == 2 && np != 1)
        return 5; // for now, only single point supported for this writer

    // Write each dataset as 1D: data[0*nt + it]
    for (size_t k = 0; k < res->ndatasets; ++k)
    {
        const FieldDataset *ds = &res->datasets[k];
        if (!ds->data)
            return 6;

        // data layout in the timeseries calculator is [p][t] with t fastest
        const float *v = ds->data; // p=0 slice

        if (diag_h5_write_field_1d(path, ds->name, ds->units, ds->long_name,
                                   0.0, 0, v, t->n, &axis_t) != 0)
            return 7;
    }

    return 0;
}

int field_diag_h5_write_split(const char *prefix, const FieldResult *res)
{
    if (!prefix || !res) return 1;
    if (!(res->naxes == 2 || res->naxes == 1)) return 2;

    /* Expect time axis in axes[0] */
    const FieldAxis *t = &res->axes[0];
    if (!t->values || t->n < 2) return 3;

    /* Build axis description (diag_h5_write_field_1d will write AXIS/AXIS1) */
    DiagAxis1D axis_t = {
        .name = t->name ? t->name : "t",
        .long_name = "t",
        .units = t->units ? t->units : "fs",
        .vmin = t->values[0],
        .vmax = t->values[t->n - 1]
    };

    /* Determine if this is a "single point timeseries" stored as 2D [p][t] */
    size_t np = 1;
    if (res->naxes == 2)
    {
        np = res->axes[1].n;
        /* For now only support single point if naxes==2 */
        if (np != 1) return 4;
    }

    /* Write one file per dataset */
    for (size_t k = 0; k < res->ndatasets; ++k)
    {
        const FieldDataset *ds = &res->datasets[k];
        if (!ds->data || !ds->name) return 5;

        char path[512];
        /* Example: prefix="ts" -> "ts_Ex.h5" */
        snprintf(path, sizeof(path), "%s_%s.h5", prefix, ds->name);

        /* For naxes==2 (single point), data layout is [p][t] with t fastest; use p=0 slice */
        const float *v = ds->data;

        int rc = diag_h5_write_field_1d(path,
                                        ds->name,      /* dataset name inside file */
                                        ds->units,
                                        ds->long_name,
                                        0.0, 0,        /* TIME/ITER (adjust if you want) */
                                        v,
                                        t->n,
                                        &axis_t);
        if (rc != 0) return 6;
    }

    return 0;
}
