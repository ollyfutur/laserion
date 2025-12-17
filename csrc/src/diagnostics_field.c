#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "diagnostics_field.h"
#include "core.h" /* LaserPulse_E / LaserPulse_A */

/* --- helpers --- */

static int axis_to_index(DiagAxis a)
{
    if (a == DIAG_AXIS_X)
        return 0;
    if (a == DIAG_AXIS_Y)
        return 1;
    if (a == DIAG_AXIS_Z)
        return 2;
    return -1;
}

static void set_coord(double r[3], DiagAxis axis, double value_um)
{
    int i = axis_to_index(axis);
    if (i >= 0)
        r[i] = value_um;
}

static double get_component(const double v[3], DiagComponent c)
{
    return v[(int)c];
}

static int eval_field_component(const LaserPulse *p, DiagFieldKind kind, DiagComponent comp,
                                double t_fs, const double r_um[3], double *out_value)
{
    double v[3];
    if (kind == DIAG_FIELD_E)
    {
        LaserPulse_E(p, t_fs, r_um, v);
    }
    else
    {
        LaserPulse_A(p, t_fs, r_um, v);
    }
    *out_value = get_component(v, comp);
    return 0;
}

/* --- diagnostics --- */

int diag_run_time_series(const DiagContext *ctx, const DiagTimeSeries *spec)
{
    if (!ctx || !spec || !ctx->pulse || !spec->csv_path)
        return 1;
    if (!(spec->nt >= 2) || !(spec->tmax_fs > spec->tmin_fs))
        return 2;

    FILE *fp = fopen(spec->csv_path, "w");
    if (!fp)
        return 3;

    fprintf(fp, "t_fs,Ex,Ey,Ez,Ax,Ay,Az\n");

    const double dt = (spec->tmax_fs - spec->tmin_fs) / (double)(spec->nt - 1);

    double r[3] = {spec->x_um, spec->y_um, spec->z_um};

    for (size_t it = 0; it < spec->nt; ++it)
    {
        double t = spec->tmin_fs + (double)it * dt;

        double E[3], A[3];
        LaserPulse_E(ctx->pulse, t, r, E);
        LaserPulse_A(ctx->pulse, t, r, A);

        fprintf(fp, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                t, E[0], E[1], E[2], A[0], A[1], A[2]);
    }

    fclose(fp);
    return 0;
}

int diag_run_lineout_1d(const DiagContext *ctx, const DiagLineout1D *spec)
{
    if (!ctx || !spec || !ctx->pulse || !spec->csv_path)
        return 1;
    if (!(spec->ns >= 2) || !(spec->smax_um > spec->smin_um))
        return 2;
    if (axis_to_index(spec->axis) < 0)
        return 2;

    FILE *fp = fopen(spec->csv_path, "w");
    if (!fp)
        return 3;

    fprintf(fp, "s_um,value\n");

    const double ds = (spec->smax_um - spec->smin_um) / (double)(spec->ns - 1);

    double r[3] = {spec->x_um, spec->y_um, spec->z_um};

    for (size_t i = 0; i < spec->ns; ++i)
    {
        double s = spec->smin_um + (double)i * ds;
        set_coord(r, spec->axis, s);

        double val = 0.0;
        eval_field_component(ctx->pulse, spec->field, spec->comp, spec->t_fs, r, &val);

        fprintf(fp, "%.17g,%.17g\n", s, val);
    }

    fclose(fp);
    return 0;
}

int diag_run_slice_2d(const DiagContext *ctx, const DiagSlice2D *spec)
{
    if (!ctx || !spec || !ctx->pulse || !spec->csv_path)
        return 1;
    if (!(spec->nu >= 2) || !(spec->nv >= 2))
        return 2;
    if (!(spec->umax_um > spec->umin_um) || !(spec->vmax_um > spec->vmin_um))
        return 2;

    int iu = axis_to_index(spec->axis_u);
    int iv = axis_to_index(spec->axis_v);
    int ifx = axis_to_index(spec->axis_fixed);
    if (iu < 0 || iv < 0 || ifx < 0)
        return 2;
    if (iu == iv || iu == ifx || iv == ifx)
        return 2;

    FILE *fp = fopen(spec->csv_path, "w");
    if (!fp)
        return 3;

    fprintf(fp, "u_um,v_um,value\n");

    const double du = (spec->umax_um - spec->umin_um) / (double)(spec->nu - 1);
    const double dv = (spec->vmax_um - spec->vmin_um) / (double)(spec->nv - 1);

    double r[3] = {0.0, 0.0, 0.0};
    set_coord(r, spec->axis_fixed, spec->fixed_um);

    for (size_t j = 0; j < spec->nv; ++j)
    {
        double v = spec->vmin_um + (double)j * dv;
        set_coord(r, spec->axis_v, v);

        for (size_t i = 0; i < spec->nu; ++i)
        {
            double u = spec->umin_um + (double)i * du;
            set_coord(r, spec->axis_u, u);

            double val = 0.0;
            eval_field_component(ctx->pulse, spec->field, spec->comp, spec->t_fs, r, &val);

            fprintf(fp, "%.17g,%.17g,%.17g\n", u, v, val);
        }
    }

    fclose(fp);
    return 0;
}
