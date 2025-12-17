// field_diag_h5.c
#include "field_diag_h5.h"
#include "diag_h5.h"
#include <stdlib.h>

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
    if (diag_h5_write_axis_1d(path, "AXIS/AXIS1", &axis_t, t->n, t->values) != 0)
        return 4;

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

    // Optionally store the point coords (if present)
    if (res->has_points)
    {
        DiagAxis1D axis_dummy = {
            .name = "p", .long_name = "p", .units = "1", .vmin = 0.0, .vmax = 0.0};
        const float x = res->points_x_um[0];
        const float y = res->points_y_um[0];
        const float z = res->points_z_um[0];
        diag_h5_write_field_1d(path, "points/x_um", "um", "x", 0.0, 0, &x, 1, &axis_dummy);
        diag_h5_write_field_1d(path, "points/y_um", "um", "y", 0.0, 0, &y, 1, &axis_dummy);
        diag_h5_write_field_1d(path, "points/z_um", "um", "z", 0.0, 0, &z, 1, &axis_dummy);
    }

    return 0;
}
