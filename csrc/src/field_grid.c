#include "field_grid.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "diag_grid.h" /* dg_run_1d/2d */

/* ----------------------------- validation -------------------------------- */

static int axis_is_valid(FG_Axis id)
{
    return (id == FG_T || id == FG_X || id == FG_Y || id == FG_Z);
}

int fg_validate_axis(const FG_AxisSpec *a)
{
    if (!a) return 1;
    if (!axis_is_valid(a->id)) return 2;

    if (a->kind == FG_AXIS_LINSPACE)
    {
        if (a->n < 2) return 3;
        return 0;
    }
    if (a->kind == FG_AXIS_VALUES)
    {
        if (!a->values) return 4;
        if (a->n_values < 2) return 5;
        return 0;
    }
    return 6;
}

int fg_validate_options(const FG_Options *opt)
{
    if (!opt) return 1;

    const uint32_t qmask = opt->quantity_mask;
    const uint32_t cmask = opt->component_mask;

    if ((qmask & ((uint32_t)FG_Q_E | (uint32_t)FG_Q_A)) == 0u) return 2;
    if ((cmask & (uint32_t)FG_C_ALL) == 0u) return 3;
    if ((cmask & ~(uint32_t)FG_C_ALL) != 0u) return 4;

    if (!(opt->write_mode == FG_WRITE_SINGLE_FILE || opt->write_mode == FG_WRITE_SPLIT_FILES))
        return 5;

    return 0;
}

int fg_validate_request_1d(const FG_Request1D *r, const FG_Options *opt)
{
    if (!r || !opt) return 1;
    int rc = fg_validate_options(opt);
    if (rc) return 10 + rc;
    rc = fg_validate_axis(&r->a1);
    if (rc) return 20 + rc;
    return 0;
}

int fg_validate_request_2d(const FG_Request2D *r, const FG_Options *opt)
{
    if (!r || !opt) return 1;
    int rc = fg_validate_options(opt);
    if (rc) return 10 + rc;
    rc = fg_validate_axis(&r->a1);
    if (rc) return 20 + rc;
    rc = fg_validate_axis(&r->a2);
    if (rc) return 30 + rc;
    if (r->a1.id == r->a2.id) return 40;
    return 0;
}

/* --------------------------- mapping FG -> DG ---------------------------- */

static DG_AxisID map_axis_id(FG_Axis a)
{
    switch (a)
    {
        case FG_T: return DG_AXIS_T;
        case FG_X: return DG_AXIS_X;
        case FG_Y: return DG_AXIS_Y;
        case FG_Z: return DG_AXIS_Z;
        default:   return DG_AXIS_T;
    }
}

static DG_AxisKind map_axis_kind(FG_AxisKind k)
{
    switch (k)
    {
        case FG_AXIS_LINSPACE: return DG_AXIS_LINSPACE;
        case FG_AXIS_VALUES:   return DG_AXIS_VALUES;
        default:               return DG_AXIS_LINSPACE;
    }
}

static DG_AxisSpec map_axis_spec(const FG_AxisSpec *a)
{
    DG_AxisSpec d;
    memset(&d, 0, sizeof(d));
    d.id = map_axis_id(a->id);
    d.kind = map_axis_kind(a->kind);
    d.name = a->name;
    d.units = a->units;
    d.min = a->min;
    d.max = a->max;
    d.n = a->n;
    d.values = a->values;
    d.n_values = a->n_values;
    return d;
}

static DG_FixedCoords map_fixed_1d(const FG_Request1D *r)
{
    DG_FixedCoords c;
    c.t_fs = r->t0_fs;
    c.x_um = r->x0_um;
    c.y_um = r->y0_um;
    c.z_um = r->z0_um;
    return c;
}

static DG_FixedCoords map_fixed_2d(const FG_Request2D *r)
{
    DG_FixedCoords c;
    c.t_fs = r->t0_fs;
    c.x_um = r->x0_um;
    c.y_um = r->y0_um;
    c.z_um = r->z0_um;
    return c;
}

/* --------------------------- dataset selection --------------------------- */

typedef enum FG_InternalDatasetKind
{
    FG_DS_EX = 0,
    FG_DS_EY = 1,
    FG_DS_EZ = 2,
    FG_DS_AX = 3,
    FG_DS_AY = 4,
    FG_DS_AZ = 5
} FG_InternalDatasetKind;

typedef struct FG_EvalCtx
{
    const struct LaserPulse *pulse;
    FG_InternalDatasetKind *kinds; /* length ndatasets, maps output index -> dataset kind */
} FG_EvalCtx;

static int fg_eval_point(const DG_FixedCoords *coords, void *ctx_void, float *out, size_t ndatasets)
{
    FG_EvalCtx *ctx = (FG_EvalCtx *)ctx_void;

    const double t = coords->t_fs;
    const double r[3] = {coords->x_um, coords->y_um, coords->z_um};

    /* Compute E and/or A only if needed at this point */
    int need_E = 0, need_A = 0;
    for (size_t d = 0; d < ndatasets; ++d)
    {
        const FG_InternalDatasetKind k = ctx->kinds[d];
        if (k == FG_DS_EX || k == FG_DS_EY || k == FG_DS_EZ) need_E = 1;
        if (k == FG_DS_AX || k == FG_DS_AY || k == FG_DS_AZ) need_A = 1;
    }

    double E[3] = {0.0, 0.0, 0.0};
    double A[3] = {0.0, 0.0, 0.0};

    if (need_E) LaserPulse_E(ctx->pulse, t, r, E);
    if (need_A) LaserPulse_A(ctx->pulse, t, r, A);

    for (size_t d = 0; d < ndatasets; ++d)
    {
        switch (ctx->kinds[d])
        {
            case FG_DS_EX: out[d] = (float)E[0]; break;
            case FG_DS_EY: out[d] = (float)E[1]; break;
            case FG_DS_EZ: out[d] = (float)E[2]; break;
            case FG_DS_AX: out[d] = (float)A[0]; break;
            case FG_DS_AY: out[d] = (float)A[1]; break;
            case FG_DS_AZ: out[d] = (float)A[2]; break;
            default:       out[d] = (float)NAN;  break;
        }
    }

    return 0;
}

static size_t count_selected(const FG_Options *opt)
{
    size_t n = 0;

    const int wantE = ((opt->quantity_mask & (uint32_t)FG_Q_E) != 0u);
    const int wantA = ((opt->quantity_mask & (uint32_t)FG_Q_A) != 0u);

    if (wantE)
    {
        if (opt->component_mask & (uint32_t)FG_C_X) ++n;
        if (opt->component_mask & (uint32_t)FG_C_Y) ++n;
        if (opt->component_mask & (uint32_t)FG_C_Z) ++n;
    }
    if (wantA)
    {
        if (opt->component_mask & (uint32_t)FG_C_X) ++n;
        if (opt->component_mask & (uint32_t)FG_C_Y) ++n;
        if (opt->component_mask & (uint32_t)FG_C_Z) ++n;
    }
    return n;
}

static void fill_dataset_desc(DG_DatasetDesc *ds,
                              FG_InternalDatasetKind kind,
                              char *name_buf,
                              size_t name_bufsz,
                              char *suffix_buf,
                              size_t suffix_bufsz)
{
    /* Dataset names and suffixes */
    const char *units = NULL;

    switch (kind)
    {
        case FG_DS_EX: snprintf(name_buf, name_bufsz, "Ex"); units = "GV/m"; break;
        case FG_DS_EY: snprintf(name_buf, name_bufsz, "Ey"); units = "GV/m"; break;
        case FG_DS_EZ: snprintf(name_buf, name_bufsz, "Ez"); units = "GV/m"; break;
        case FG_DS_AX: snprintf(name_buf, name_bufsz, "Ax"); units = "a.u."; break;
        case FG_DS_AY: snprintf(name_buf, name_bufsz, "Ay"); units = "a.u."; break;
        case FG_DS_AZ: snprintf(name_buf, name_bufsz, "Az"); units = "a.u."; break;
        default:       snprintf(name_buf, name_bufsz, "unknown"); units = ""; break;
    }

    snprintf(suffix_buf, suffix_bufsz, "_%s.h5", name_buf);

    ds->dset_name = name_buf;
    ds->label = name_buf;
    ds->units = units;
    ds->file_suffix = suffix_buf;
}

/* Build dataset list and evaluation map.
 * Caller owns returned arrays and must free:
 *   - desc
 *   - kinds
 *   - name_store (array of char*)
 *   - suffix_store (array of char*)
 */
static int build_datasets(const FG_Options *opt,
                          DG_DatasetDesc **desc_out,
                          FG_InternalDatasetKind **kinds_out,
                          char ***name_store_out,
                          char ***suffix_store_out,
                          size_t *ndatasets_out)
{
    *desc_out = NULL;
    *kinds_out = NULL;
    *name_store_out = NULL;
    *suffix_store_out = NULL;
    *ndatasets_out = 0;

    const size_t nd = count_selected(opt);
    if (nd == 0) return 1;

    DG_DatasetDesc *desc = (DG_DatasetDesc *)calloc(nd, sizeof(DG_DatasetDesc));
    FG_InternalDatasetKind *kinds = (FG_InternalDatasetKind *)calloc(nd, sizeof(FG_InternalDatasetKind));
    char **name_store = (char **)calloc(nd, sizeof(char *));
    char **suffix_store = (char **)calloc(nd, sizeof(char *));
    if (!desc || !kinds || !name_store || !suffix_store)
    {
        free(desc); free(kinds); free(name_store); free(suffix_store);
        return 2;
    }

    for (size_t i = 0; i < nd; ++i)
    {
        name_store[i] = (char *)calloc(16, 1);
        suffix_store[i] = (char *)calloc(32, 1);
        if (!name_store[i] || !suffix_store[i])
        {
            for (size_t k = 0; k <= i; ++k)
            {
                free(name_store[k]);
                free(suffix_store[k]);
            }
            free(desc); free(kinds); free(name_store); free(suffix_store);
            return 3;
        }
    }

    size_t j = 0;
    const int wantE = ((opt->quantity_mask & (uint32_t)FG_Q_E) != 0u);
    const int wantA = ((opt->quantity_mask & (uint32_t)FG_Q_A) != 0u);

    if (wantE)
    {
        if (opt->component_mask & (uint32_t)FG_C_X) kinds[j++] = FG_DS_EX;
        if (opt->component_mask & (uint32_t)FG_C_Y) kinds[j++] = FG_DS_EY;
        if (opt->component_mask & (uint32_t)FG_C_Z) kinds[j++] = FG_DS_EZ;
    }
    if (wantA)
    {
        if (opt->component_mask & (uint32_t)FG_C_X) kinds[j++] = FG_DS_AX;
        if (opt->component_mask & (uint32_t)FG_C_Y) kinds[j++] = FG_DS_AY;
        if (opt->component_mask & (uint32_t)FG_C_Z) kinds[j++] = FG_DS_AZ;
    }

    for (size_t i = 0; i < nd; ++i)
        fill_dataset_desc(&desc[i], kinds[i], name_store[i], 16, suffix_store[i], 32);

    *desc_out = desc;
    *kinds_out = kinds;
    *name_store_out = name_store;
    *suffix_store_out = suffix_store;
    *ndatasets_out = nd;

    return 0;
}

static void free_datasets(DG_DatasetDesc *desc,
                          FG_InternalDatasetKind *kinds,
                          char **name_store,
                          char **suffix_store,
                          size_t nd)
{
    (void)desc;
    free(kinds);

    if (name_store)
    {
        for (size_t i = 0; i < nd; ++i) free(name_store[i]);
        free(name_store);
    }
    if (suffix_store)
    {
        for (size_t i = 0; i < nd; ++i) free(suffix_store[i]);
        free(suffix_store);
    }
    free(desc);
}

/* ------------------------------- public run ------------------------------ */

int fg_run_1d(const struct LaserPulse *pulse,
              const FG_Request1D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm)
{
    if (!pulse || !req || !opt || !path_or_prefix) return 1;

    const int vrc = fg_validate_request_1d(req, opt);
    if (vrc) return 100 + vrc;

    if (opt->write_mode == FG_WRITE_SINGLE_FILE)
    {
        /* Not supported with current diag_h5 API (would overwrite with TRUNC). */
        return 2;
    }

    DG_Request1D dreq;
    dreq.a1 = map_axis_spec(&req->a1);
    dreq.fixed = map_fixed_1d(req);

    DG_RunOptions ropt = dg_default_run_options();
    ropt.root_rank = opt->root_rank;
    ropt.write_time_iter_0 = 1;

    DG_DatasetDesc *desc = NULL;
    FG_InternalDatasetKind *kinds = NULL;
    char **name_store = NULL;
    char **suffix_store = NULL;
    size_t nd = 0;

    int rc = build_datasets(opt, &desc, &kinds, &name_store, &suffix_store, &nd);
    if (rc) return 200 + rc;

    FG_EvalCtx ctx;
    ctx.pulse = pulse;
    ctx.kinds = kinds;

    rc = dg_run_1d(&dreq, desc, nd, fg_eval_point, &ctx, &ropt, path_or_prefix, comm);

    free_datasets(desc, kinds, name_store, suffix_store, nd);
    return rc;
}

int fg_run_2d(const struct LaserPulse *pulse,
              const FG_Request2D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm)
{
    if (!pulse || !req || !opt || !path_or_prefix) return 1;

    const int vrc = fg_validate_request_2d(req, opt);
    if (vrc) return 100 + vrc;

    if (opt->write_mode == FG_WRITE_SINGLE_FILE)
    {
        /* Not supported with current diag_h5 API (would overwrite with TRUNC). */
        return 2;
    }

    DG_Request2D dreq;
    dreq.a1 = map_axis_spec(&req->a1);
    dreq.a2 = map_axis_spec(&req->a2);
    dreq.fixed = map_fixed_2d(req);

    DG_RunOptions ropt = dg_default_run_options();
    ropt.root_rank = opt->root_rank;
    ropt.write_time_iter_0 = 1;

    DG_DatasetDesc *desc = NULL;
    FG_InternalDatasetKind *kinds = NULL;
    char **name_store = NULL;
    char **suffix_store = NULL;
    size_t nd = 0;

    int rc = build_datasets(opt, &desc, &kinds, &name_store, &suffix_store, &nd);
    if (rc) return 200 + rc;

    FG_EvalCtx ctx;
    ctx.pulse = pulse;
    ctx.kinds = kinds;

    rc = dg_run_2d(&dreq, desc, nd, fg_eval_point, &ctx, &ropt, path_or_prefix, comm);

    free_datasets(desc, kinds, name_store, suffix_store, nd);
    return rc;
}

