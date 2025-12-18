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

#include "field_diag_h5.h"
#include "diag_h5.h"

#include <stdio.h>
#include <string.h>

int field_diag_h5_write_split(const char *prefix, const FieldResult *res)
{
    if (!prefix || !res)
        return 1;

    /* We expect timeseries results to have time in axes[0] */
    if (res->naxes < 1)
        return 2;

    const FieldAxis *t = &res->axes[0];
    if (!t->values || t->n < 2)
        return 3;

    /* Axis1 = time */
    DiagAxis1D axis_t = {
        .name = t->name ? t->name : "t",
        .long_name = "t",
        .units = t->units ? t->units : "fs",
        .vmin = t->values[0],
        .vmax = t->values[t->n - 1]};

    /* Determine whether we have a point axis */
    size_t np = 1;
    int has_p_axis = 0;

    if (res->naxes == 2)
    {
        if (res->axes[1].id != FIELD_AXIS_P)
            return 4; /* safeguard */
        np = res->axes[1].n;
        if (np < 1)
            return 5;
        has_p_axis = 1;
    }
    else if (res->naxes != 1)
    {
        return 6; /* this adapter only supports 1D or 2D results */
    }

    /* Axis2 = point index (only used if np>1) */
    DiagAxis1D axis_p = {
        .name = "p",
        .long_name = "p",
        .units = "1",
        .vmin = 0.0,
        .vmax = (np > 0) ? (double)(np - 1) : 0.0};

    const size_t nt = t->n;

    for (size_t k = 0; k < res->ndatasets; ++k)
    {
        const FieldDataset *ds = &res->datasets[k];
        if (!ds->data || !ds->name)
            return 7;

        char path[512];
        snprintf(path, sizeof(path), "%s_%s.h5", prefix, ds->name);

        if (!has_p_axis || np == 1)
        {
            /* Write 1D: p=0 slice, length nt */
            const float *v = ds->data; /* [p][t] with p=0 -> contiguous [t] */
            int rc = diag_h5_write_field_1d(path,
                                            ds->name, ds->units, ds->long_name,
                                            0.0, 0,
                                            v, nt,
                                            &axis_t);
            if (rc != 0)
                return 8;
        }
        else
        {
            /* Write 2D: shape [np, nt] where AXIS1 is time (nt, fastest), AXIS2 is p (np, slow)
               Memory is row-major [p][t] which matches diag_h5_write_field_2d contract:
                 data size = n2*n1, n1 fastest axis (AXIS1)
               So: n1 = nt, n2 = np. */
            int rc = diag_h5_write_field_2d(path,
                                            ds->name, ds->units, ds->long_name,
                                            0.0, 0,
                                            ds->data,
                                            nt, np,
                                            &axis_t, &axis_p);
            if (rc != 0)
                return 9;
        }
    }

    return 0;
}
