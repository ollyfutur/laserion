#include "field_grid.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#include <hdf5.h>

#include "diag_grid.h" /* dg_run_1d/2d */

/* If LaserPulse_E/A are declared in a different header in your project,
 * include it here instead of forward-declaring.
 */
struct LaserPulse;
void LaserPulse_E(const struct LaserPulse *p, double t_fs, const double r_um[3], double E_out[3]);
void LaserPulse_A(const struct LaserPulse *p, double t_fs, const double r_um[3], double A_out[3]);

/* ----------------------------- validation -------------------------------- */

static int axis_is_valid(FG_Axis id)
{
    return (id == FG_T || id == FG_X || id == FG_Y || id == FG_Z);
}

int fg_validate_axis(const FG_AxisSpec *a)
{
    if (!a)
        return 1;
    if (!axis_is_valid(a->id))
        return 2;

    if (a->kind == FG_AXIS_LINSPACE)
    {
        if (a->n < 2)
            return 3;
        return 0;
    }
    if (a->kind == FG_AXIS_VALUES)
    {
        if (!a->values)
            return 4;
        if (a->n_values < 2)
            return 5;
        return 0;
    }
    return 6;
}

int fg_validate_options(const FG_Options *opt)
{
    if (!opt)
        return 1;

    const uint32_t qmask = opt->quantity_mask;
    const uint32_t cmask = opt->component_mask;

    if ((qmask & ((uint32_t)FG_Q_E | (uint32_t)FG_Q_A)) == 0u)
        return 2;
    if ((cmask & (uint32_t)FG_C_ALL) == 0u)
        return 3;
    if ((cmask & ~(uint32_t)FG_C_ALL) != 0u)
        return 4;

    if (!(opt->write_mode == FG_WRITE_SINGLE_FILE || opt->write_mode == FG_WRITE_SPLIT_FILES))
        return 5;

    return 0;
}

int fg_validate_request_1d(const FG_Request1D *r, const FG_Options *opt)
{
    if (!r || !opt)
        return 1;
    int rc = fg_validate_options(opt);
    if (rc)
        return 10 + rc;
    rc = fg_validate_axis(&r->a1);
    if (rc)
        return 20 + rc;
    return 0;
}

int fg_validate_request_2d(const FG_Request2D *r, const FG_Options *opt)
{
    if (!r || !opt)
        return 1;
    int rc = fg_validate_options(opt);
    if (rc)
        return 10 + rc;
    rc = fg_validate_axis(&r->a1);
    if (rc)
        return 20 + rc;
    rc = fg_validate_axis(&r->a2);
    if (rc)
        return 30 + rc;
    if (r->a1.id == r->a2.id)
        return 40;
    return 0;
}

/* --------------------------- mapping FG -> DG ---------------------------- */

static DG_AxisID map_axis_id(FG_Axis a)
{
    switch (a)
    {
    case FG_T:
        return DG_AXIS_T;
    case FG_X:
        return DG_AXIS_X;
    case FG_Y:
        return DG_AXIS_Y;
    case FG_Z:
        return DG_AXIS_Z;
    default:
        return DG_AXIS_T;
    }
}

static DG_AxisKind map_axis_kind(FG_AxisKind k)
{
    switch (k)
    {
    case FG_AXIS_LINSPACE:
        return DG_AXIS_LINSPACE;
    case FG_AXIS_VALUES:
        return DG_AXIS_VALUES;
    default:
        return DG_AXIS_LINSPACE;
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

/* ------------------------- pulse-backed evaluator ------------------------ */

typedef struct FG_EvalCtx
{
    const struct LaserPulse *pulse;
    FG_InternalDatasetKind *kinds; /* length ndatasets, maps output index -> dataset kind */
} FG_EvalCtx;

static int fg_eval_point_pulse(const DG_FixedCoords *coords, void *ctx_void, float *out, size_t ndatasets)
{
    FG_EvalCtx *ctx = (FG_EvalCtx *)ctx_void;

    const double t = coords->t_fs;
    const double r[3] = {coords->x_um, coords->y_um, coords->z_um};

    /* Compute E and/or A only if needed at this point */
    int need_E = 0, need_A = 0;
    for (size_t d = 0; d < ndatasets; ++d)
    {
        const FG_InternalDatasetKind k = ctx->kinds[d];
        if (k == FG_DS_EX || k == FG_DS_EY || k == FG_DS_EZ)
            need_E = 1;
        if (k == FG_DS_AX || k == FG_DS_AY || k == FG_DS_AZ)
            need_A = 1;
    }

    double E[3] = {0.0, 0.0, 0.0};
    double A[3] = {0.0, 0.0, 0.0};

    if (need_E)
        LaserPulse_E(ctx->pulse, t, r, E);
    if (need_A)
        LaserPulse_A(ctx->pulse, t, r, A);

    for (size_t d = 0; d < ndatasets; ++d)
    {
        switch (ctx->kinds[d])
        {
        case FG_DS_EX:
            out[d] = (float)E[0];
            break;
        case FG_DS_EY:
            out[d] = (float)E[1];
            break;
        case FG_DS_EZ:
            out[d] = (float)E[2];
            break;
        case FG_DS_AX:
            out[d] = (float)A[0];
            break;
        case FG_DS_AY:
            out[d] = (float)A[1];
            break;
        case FG_DS_AZ:
            out[d] = (float)A[2];
            break;
        default:
            out[d] = (float)NAN;
            break;
        }
    }

    return 0;
}

/* -------------------------- dataset list building ------------------------ */

static size_t count_selected(const FG_Options *opt)
{
    size_t n = 0;

    const int wantE = ((opt->quantity_mask & (uint32_t)FG_Q_E) != 0u);
    const int wantA = ((opt->quantity_mask & (uint32_t)FG_Q_A) != 0u);

    if (wantE)
    {
        if (opt->component_mask & (uint32_t)FG_C_X)
            ++n;
        if (opt->component_mask & (uint32_t)FG_C_Y)
            ++n;
        if (opt->component_mask & (uint32_t)FG_C_Z)
            ++n;
    }
    if (wantA)
    {
        if (opt->component_mask & (uint32_t)FG_C_X)
            ++n;
        if (opt->component_mask & (uint32_t)FG_C_Y)
            ++n;
        if (opt->component_mask & (uint32_t)FG_C_Z)
            ++n;
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
    const char *units = NULL;

    switch (kind)
    {
    case FG_DS_EX:
        snprintf(name_buf, name_bufsz, "Ex");
        units = "GV/m";
        break;
    case FG_DS_EY:
        snprintf(name_buf, name_bufsz, "Ey");
        units = "GV/m";
        break;
    case FG_DS_EZ:
        snprintf(name_buf, name_bufsz, "Ez");
        units = "GV/m";
        break;
    case FG_DS_AX:
        snprintf(name_buf, name_bufsz, "Ax");
        units = "GV/m fs";
        break;
    case FG_DS_AY:
        snprintf(name_buf, name_bufsz, "Ay");
        units = "GV/m fs";
        break;
    case FG_DS_AZ:
        snprintf(name_buf, name_bufsz, "Az");
        units = "GV/m fs";
        break;
    default:
        snprintf(name_buf, name_bufsz, "unknown");
        units = "";
        break;
    }

    snprintf(suffix_buf, suffix_bufsz, "_%s.h5", name_buf);

    ds->dset_name = name_buf;
    ds->label = name_buf;
    ds->units = units;
    ds->file_suffix = suffix_buf;
}

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
    if (nd == 0)
        return 1;

    DG_DatasetDesc *desc = (DG_DatasetDesc *)calloc(nd, sizeof(DG_DatasetDesc));
    FG_InternalDatasetKind *kinds = (FG_InternalDatasetKind *)calloc(nd, sizeof(FG_InternalDatasetKind));
    char **name_store = (char **)calloc(nd, sizeof(char *));
    char **suffix_store = (char **)calloc(nd, sizeof(char *));
    if (!desc || !kinds || !name_store || !suffix_store)
    {
        free(desc);
        free(kinds);
        free(name_store);
        free(suffix_store);
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
            free(desc);
            free(kinds);
            free(name_store);
            free(suffix_store);
            return 3;
        }
    }

    size_t j = 0;
    const int wantE = ((opt->quantity_mask & (uint32_t)FG_Q_E) != 0u);
    const int wantA = ((opt->quantity_mask & (uint32_t)FG_Q_A) != 0u);

    if (wantE)
    {
        if (opt->component_mask & (uint32_t)FG_C_X)
            kinds[j++] = FG_DS_EX;
        if (opt->component_mask & (uint32_t)FG_C_Y)
            kinds[j++] = FG_DS_EY;
        if (opt->component_mask & (uint32_t)FG_C_Z)
            kinds[j++] = FG_DS_EZ;
    }
    if (wantA)
    {
        if (opt->component_mask & (uint32_t)FG_C_X)
            kinds[j++] = FG_DS_AX;
        if (opt->component_mask & (uint32_t)FG_C_Y)
            kinds[j++] = FG_DS_AY;
        if (opt->component_mask & (uint32_t)FG_C_Z)
            kinds[j++] = FG_DS_AZ;
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
        for (size_t i = 0; i < nd; ++i)
            free(name_store[i]);
        free(name_store);
    }
    if (suffix_store)
    {
        for (size_t i = 0; i < nd; ++i)
            free(suffix_store[i]);
        free(suffix_store);
    }
    free(desc);
}

/* ----------------------------- cache support ----------------------------- */

typedef struct FG_CacheDset
{
    hid_t f;
    hid_t dset;
    int nd; /* 2 or 3 */
    int t_n, ax1_n, ax2_n;
    double t_min, dt;
    double ax1_min, dx1;
    double ax2_min, dx2;
    char axis2[8]; /* "x"|"y"|"z" : dataset dim 2 */
    char axis3[8]; /* "x"|"y"|"z" : dataset dim 3 (if nd==3) */
} FG_CacheDset;

typedef struct FG_CacheCtx
{
    FG_InternalDatasetKind *kinds; /* output index -> dataset kind */
    FG_CacheDset d[6];
    int opened[6];
} FG_CacheCtx;

static int h5_read_attr_double(hid_t obj, const char *name, double *out)
{
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0)
        return 1;
    herr_t st = H5Aread(a, H5T_NATIVE_DOUBLE, out);
    H5Aclose(a);
    return (st < 0) ? 2 : 0;
}

static int h5_read_attr_int(hid_t obj, const char *name, int *out)
{
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0)
        return 1;
    herr_t st = H5Aread(a, H5T_NATIVE_INT, out);
    H5Aclose(a);
    return (st < 0) ? 2 : 0;
}

/* Reads a scalar fixed-length string attribute written by field_cache.c (h5_write_attr_string). */
static int h5_read_attr_string(hid_t obj, const char *name, char *buf, size_t bufsz)
{
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0)
        return 1;

    hid_t at = H5Aget_type(a);
    if (at < 0)
    {
        H5Aclose(a);
        return 2;
    }

    size_t sz = (size_t)H5Tget_size(at);
    if (sz == 0 || sz > 1024)
    {
        H5Tclose(at);
        H5Aclose(a);
        return 3;
    }

    char tmp[1024];
    memset(tmp, 0, sizeof(tmp));

    herr_t st = H5Aread(a, at, tmp);

    H5Tclose(at);
    H5Aclose(a);

    if (st < 0)
        return 4;

    tmp[sz < sizeof(tmp) ? sz : sizeof(tmp) - 1] = '\0';
    snprintf(buf, bufsz, "%s", tmp);
    return 0;
}

static double coord_for_axis(const DG_FixedCoords *c, const char *ax)
{
    if (!ax || !ax[0])
        return 0.0;
    const char a = (char)tolower((unsigned char)ax[0]);
    if (a == 'x')
        return c->x_um;
    if (a == 'y')
        return c->y_um;
    if (a == 'z')
        return c->z_um;
    return 0.0;
}

static int index_from_lin(double v, double vmin, double dv, int n)
{
    if (!(dv > 0.0) || n <= 0)
        return 0;
    const double x = (v - vmin) / dv;
    long i = lround(x);
    if (i < 0)
        i = 0;
    if (i > (long)(n - 1))
        i = (long)(n - 1);
    return (int)i;
}

static int fg_cache_open_one(FG_CacheDset *cd, const char *cache_dir, const char *comp)
{
    memset(cd, 0, sizeof(*cd));
    cd->f = -1;
    cd->dset = -1;

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.h5", cache_dir, comp);

    cd->f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (cd->f < 0)
    {
        fprintf(stderr, "field_diag: could not open cache file %s\n", path);
        return 1;
    }

    char dpath[64];
    snprintf(dpath, sizeof(dpath), "/%s", comp);

    cd->dset = H5Dopen2(cd->f, dpath, H5P_DEFAULT);
    if (cd->dset < 0)
    {
        fprintf(stderr, "field_diag: could not open dataset %s in %s\n", dpath, path);
        H5Fclose(cd->f);
        cd->f = -1;
        return 2;
    }

    if (h5_read_attr_double(cd->f, "t_min_fs", &cd->t_min) != 0)
        return 3;
    if (h5_read_attr_double(cd->f, "dt_fs", &cd->dt) != 0)
        return 4;
    if (h5_read_attr_int(cd->f, "t_n", &cd->t_n) != 0)
        return 5;

    if (h5_read_attr_double(cd->f, "ax1_min_um", &cd->ax1_min) != 0)
        return 6;
    if (h5_read_attr_double(cd->f, "dx1_um", &cd->dx1) != 0)
        return 7;
    if (h5_read_attr_int(cd->f, "ax1_n", &cd->ax1_n) != 0)
        return 8;

    if (h5_read_attr_string(cd->f, "axis2", cd->axis2, sizeof(cd->axis2)) != 0)
        return 9;

    hid_t space = H5Dget_space(cd->dset);
    if (space < 0)
        return 10;
    int nd = H5Sget_simple_extent_ndims(space);
    H5Sclose(space);

    if (nd != 2 && nd != 3)
    {
        fprintf(stderr, "field_diag: cache dataset %s has ndims=%d (expected 2 or 3)\n", comp, nd);
        return 11;
    }
    cd->nd = nd;

    if (nd == 3)
    {
        if (h5_read_attr_double(cd->f, "ax2_min_um", &cd->ax2_min) != 0)
            return 12;
        if (h5_read_attr_double(cd->f, "dx2_um", &cd->dx2) != 0)
            return 13;
        if (h5_read_attr_int(cd->f, "ax2_n", &cd->ax2_n) != 0)
            return 14;
        if (h5_read_attr_string(cd->f, "axis3", cd->axis3, sizeof(cd->axis3)) != 0)
            return 15;
    }
    else
    {
        cd->ax2_min = 0.0;
        cd->dx2 = 0.0;
        cd->ax2_n = 0;
        cd->axis3[0] = '\0';
    }

    return 0;
}

static void fg_cache_close_one(FG_CacheDset *cd)
{
    if (!cd)
        return;
    if (cd->dset >= 0)
        H5Dclose(cd->dset);
    if (cd->f >= 0)
        H5Fclose(cd->f);
    cd->dset = -1;
    cd->f = -1;
}

static int fg_eval_point_cache(const DG_FixedCoords *coords, void *ctx_void, float *out, size_t ndatasets)
{
    FG_CacheCtx *ctx = (FG_CacheCtx *)ctx_void;

    for (size_t d = 0; d < ndatasets; ++d)
    {
        const FG_InternalDatasetKind k = ctx->kinds[d];
        FG_CacheDset *cd = &ctx->d[(int)k];

        /* dataset indexing order is (t, ax1, ax2) */
        const int it = index_from_lin(coords->t_fs, cd->t_min, cd->dt, cd->t_n);

        const double v1 = coord_for_axis(coords, cd->axis2);
        const int i1 = index_from_lin(v1, cd->ax1_min, cd->dx1, cd->ax1_n);

        int i2 = 0;
        if (cd->nd == 3)
        {
            const double v2 = coord_for_axis(coords, cd->axis3);
            i2 = index_from_lin(v2, cd->ax2_min, cd->dx2, cd->ax2_n);
        }

        double val = NAN;

        if (cd->nd == 2)
        {
            hsize_t start[2] = {(hsize_t)it, (hsize_t)i1};
            hsize_t count[2] = {1, 1};

            hid_t fspace = H5Dget_space(cd->dset);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

            hid_t mspace = H5Screate(H5S_SCALAR);
            herr_t st = H5Dread(cd->dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, &val);

            H5Sclose(mspace);
            H5Sclose(fspace);

            if (st < 0)
                val = NAN;
        }
        else
        {
            hsize_t start[3] = {(hsize_t)it, (hsize_t)i1, (hsize_t)i2};
            hsize_t count[3] = {1, 1, 1};

            hid_t fspace = H5Dget_space(cd->dset);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

            hid_t mspace = H5Screate(H5S_SCALAR);
            herr_t st = H5Dread(cd->dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, &val);

            H5Sclose(mspace);
            H5Sclose(fspace);

            if (st < 0)
                val = NAN;
        }

        out[d] = (float)val;
    }

    return 0;
}

static const char *kind_to_comp(FG_InternalDatasetKind k)
{
    switch (k)
    {
    case FG_DS_EX:
        return "Ex";
    case FG_DS_EY:
        return "Ey";
    case FG_DS_EZ:
        return "Ez";
    case FG_DS_AX:
        return "Ax";
    case FG_DS_AY:
        return "Ay";
    case FG_DS_AZ:
        return "Az";
    default:
        return "Ex";
    }
}

/* ------------------------------- public run ------------------------------ */

int fg_run_1d(const struct LaserPulse *pulse,
              const FG_Request1D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm)
{
    if (!pulse || !req || !opt || !path_or_prefix)
        return 1;

    const int vrc = fg_validate_request_1d(req, opt);
    if (vrc)
        return 100 + vrc;

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
    if (rc)
        return 200 + rc;

    FG_EvalCtx ctx;
    ctx.pulse = pulse;
    ctx.kinds = kinds;

    rc = dg_run_1d(&dreq, desc, nd, fg_eval_point_pulse, &ctx, &ropt, path_or_prefix, comm);

    free_datasets(desc, kinds, name_store, suffix_store, nd);
    return rc;
}

int fg_run_2d(const struct LaserPulse *pulse,
              const FG_Request2D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm)
{
    if (!pulse || !req || !opt || !path_or_prefix)
        return 1;

    const int vrc = fg_validate_request_2d(req, opt);
    if (vrc)
        return 100 + vrc;

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
    if (rc)
        return 200 + rc;

    FG_EvalCtx ctx;
    ctx.pulse = pulse;
    ctx.kinds = kinds;

    rc = dg_run_2d(&dreq, desc, nd, fg_eval_point_pulse, &ctx, &ropt, path_or_prefix, comm);

    free_datasets(desc, kinds, name_store, suffix_store, nd);
    return rc;
}

int fg_run_1d_from_cache(const char *cache_dir,
                         const FG_Request1D *req,
                         const FG_Options *opt,
                         const char *path_or_prefix,
                         MPI_Comm comm)
{
    if (!cache_dir || !req || !opt || !path_or_prefix)
        return 1;

    const int vrc = fg_validate_request_1d(req, opt);
    if (vrc)
        return 100 + vrc;

    if (opt->write_mode == FG_WRITE_SINGLE_FILE)
        return 2;

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
    if (rc)
        return 200 + rc;

    FG_CacheCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kinds = kinds;

    for (size_t i = 0; i < nd; ++i)
    {
        const FG_InternalDatasetKind k = kinds[i];
        if (!ctx.opened[(int)k])
        {
            const char *comp = kind_to_comp(k);
            int orc = fg_cache_open_one(&ctx.d[(int)k], cache_dir, comp);
            if (orc != 0)
            {
                int rank = 0;
                MPI_Comm_rank(comm, &rank);
                if (rank == ropt.root_rank)
                    fprintf(stderr, "field_diag: failed opening cache component %s (rc=%d)\n", comp, orc);

                for (int kk = 0; kk < 6; ++kk)
                    if (ctx.opened[kk])
                        fg_cache_close_one(&ctx.d[kk]);

                free_datasets(desc, kinds, name_store, suffix_store, nd);
                return 300 + orc;
            }
            ctx.opened[(int)k] = 1;
        }
    }

    rc = dg_run_1d(&dreq, desc, nd, fg_eval_point_cache, &ctx, &ropt, path_or_prefix, comm);

    for (int kk = 0; kk < 6; ++kk)
        if (ctx.opened[kk])
            fg_cache_close_one(&ctx.d[kk]);

    free_datasets(desc, kinds, name_store, suffix_store, nd);
    return rc;
}

int fg_run_2d_from_cache(const char *cache_dir,
                         const FG_Request2D *req,
                         const FG_Options *opt,
                         const char *path_or_prefix,
                         MPI_Comm comm)
{
    if (!cache_dir || !req || !opt || !path_or_prefix)
        return 1;

    const int vrc = fg_validate_request_2d(req, opt);
    if (vrc)
        return 100 + vrc;

    if (opt->write_mode == FG_WRITE_SINGLE_FILE)
        return 2;

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
    if (rc)
        return 200 + rc;

    FG_CacheCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kinds = kinds;

    for (size_t i = 0; i < nd; ++i)
    {
        const FG_InternalDatasetKind k = kinds[i];
        if (!ctx.opened[(int)k])
        {
            const char *comp = kind_to_comp(k);
            int orc = fg_cache_open_one(&ctx.d[(int)k], cache_dir, comp);
            if (orc != 0)
            {
                int rank = 0;
                MPI_Comm_rank(comm, &rank);
                if (rank == ropt.root_rank)
                    fprintf(stderr, "field_diag: failed opening cache component %s (rc=%d)\n", comp, orc);

                for (int kk = 0; kk < 6; ++kk)
                    if (ctx.opened[kk])
                        fg_cache_close_one(&ctx.d[kk]);

                free_datasets(desc, kinds, name_store, suffix_store, nd);
                return 300 + orc;
            }
            ctx.opened[(int)k] = 1;
        }
    }

    rc = dg_run_2d(&dreq, desc, nd, fg_eval_point_cache, &ctx, &ropt, path_or_prefix, comm);

    for (int kk = 0; kk < 6; ++kk)
        if (ctx.opened[kk])
            fg_cache_close_one(&ctx.d[kk]);

    free_datasets(desc, kinds, name_store, suffix_store, nd);
    return rc;
}
