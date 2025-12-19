#include "diag_grid.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ------------------------------ utilities -------------------------------- */

DG_RunOptions dg_default_run_options(void)
{
    DG_RunOptions o;
    o.root_rank = 0;
    o.write_time_iter_0 = 1;
    return o;
}

static DiagAxisID to_diag_axis_id(DG_AxisID id)
{
    switch (id)
    {
        case DG_AXIS_T: return DIAG_AXIS_T;
        case DG_AXIS_X: return DIAG_AXIS_X;
        case DG_AXIS_Y: return DIAG_AXIS_Y;
        case DG_AXIS_Z: return DIAG_AXIS_Z;
        default:        return DIAG_AXIS_T;
    }
}

static void compute_counts_displs(int size, int n, int *counts, int *displs)
{
    const int q = (size > 0) ? (n / size) : 0;
    const int r = (size > 0) ? (n % size) : 0;
    int off = 0;
    for (int i = 0; i < size; ++i)
    {
        counts[i] = q + (i < r ? 1 : 0);
        displs[i] = off;
        off += counts[i];
    }
}

static double *materialize_axis_values(const DG_AxisSpec *a, size_t *n_out)
{
    *n_out = 0;
    if (!a) return NULL;

    if (a->kind == DG_AXIS_VALUES)
    {
        if (!a->values || a->n_values < 2) return NULL;
        const size_t n = a->n_values;
        double *v = (double *)malloc(n * sizeof(double));
        if (!v) return NULL;
        memcpy(v, a->values, n * sizeof(double));
        *n_out = n;
        return v;
    }

    /* linspace */
    if (a->n < 2) return NULL;
    const size_t n = a->n;

    double *v = (double *)malloc(n * sizeof(double));
    if (!v) return NULL;

    const double a0 = a->min;
    const double b0 = a->max;
    const double denom = (double)(n - 1);

    for (size_t i = 0; i < n; ++i)
    {
        const double s = (denom > 0.0) ? ((double)i / denom) : 0.0;
        v[i] = a0 + (b0 - a0) * s;
    }

    *n_out = n;
    return v;
}

static void apply_axis(DG_AxisID id, double val, DG_FixedCoords *c)
{
    switch (id)
    {
        case DG_AXIS_T: c->t_fs = val; break;
        case DG_AXIS_X: c->x_um = val; break;
        case DG_AXIS_Y: c->y_um = val; break;
        case DG_AXIS_Z: c->z_um = val; break;
        default: break;
    }
}

static DiagAxis make_diag_axis(const DG_AxisSpec *spec, const double *vals, size_t nvals)
{
    DiagAxis a;
    a.id        = to_diag_axis_id(spec->id);
    a.long_name = spec->name;
    a.units     = spec->units;
    a.vmin      = (nvals > 0) ? vals[0] : 0.0;
    a.vmax      = (nvals > 0) ? vals[nvals - 1] : 0.0;
    return a;
}

static DiagFixedCoords to_fixed_coords(const DG_FixedCoords *c)
{
    DiagFixedCoords f;
    f.t = c->t_fs;
    f.x = c->x_um;
    f.y = c->y_um;
    f.z = c->z_um;
    return f;
}

static void make_path(char *buf, size_t bufsz, const char *prefix, const char *suffix)
{
    snprintf(buf, bufsz, "%s%s", prefix, suffix);
}

/* ------------------------------ validation -------------------------------- */

int dg_validate_axis(const DG_AxisSpec *a)
{
    if (!a) return 1;
    if (!(a->id == DG_AXIS_T || a->id == DG_AXIS_X || a->id == DG_AXIS_Y || a->id == DG_AXIS_Z))
        return 2;

    if (a->kind == DG_AXIS_LINSPACE)
    {
        if (a->n < 2) return 3;
        return 0;
    }
    if (a->kind == DG_AXIS_VALUES)
    {
        if (!a->values || a->n_values < 2) return 4;
        return 0;
    }
    return 5;
}

int dg_validate_req_1d(const DG_Request1D *req)
{
    if (!req) return 1;
    return dg_validate_axis(&req->a1);
}

int dg_validate_req_2d(const DG_Request2D *req)
{
    if (!req) return 1;
    int rc = dg_validate_axis(&req->a1);
    if (rc) return 10 + rc;
    rc = dg_validate_axis(&req->a2);
    if (rc) return 20 + rc;
    if (req->a1.id == req->a2.id) return 30;
    return 0;
}

/* ------------------------------ writers ----------------------------------- */

static int write_1d_one(const DG_Request1D *req,
                        const DG_DatasetDesc *ds,
                        const float *data,
                        size_t n1,
                        const DiagAxis *axis1,
                        int write_time_iter_0,
                        const char *path_prefix)
{
    char path[1024];
    make_path(path, sizeof(path), path_prefix, ds->file_suffix);

    const DiagFixedCoords fixed = to_fixed_coords(&req->fixed);

    const double time_attr = write_time_iter_0 ? 0.0 : req->fixed.t_fs;
    const int    iter_attr = write_time_iter_0 ? 0    : 0;

    return diag_h5_write_grid_1d(path,
                                 ds->dset_name,
                                 ds->units,
                                 ds->label,
                                 time_attr,
                                 iter_attr,
                                 data,
                                 n1,
                                 axis1,
                                 &fixed);
}

static int write_2d_one(const DG_Request2D *req,
                        const DG_DatasetDesc *ds,
                        const float *data,
                        size_t n1,
                        size_t n2,
                        const DiagAxis *axis1,
                        const DiagAxis *axis2,
                        int write_time_iter_0,
                        const char *path_prefix)
{
    char path[1024];
    make_path(path, sizeof(path), path_prefix, ds->file_suffix);

    const DiagFixedCoords fixed = to_fixed_coords(&req->fixed);

    const double time_attr = write_time_iter_0 ? 0.0 : req->fixed.t_fs;
    const int    iter_attr = write_time_iter_0 ? 0    : 0;

    return diag_h5_write_grid_2d(path,
                                 ds->dset_name,
                                 ds->units,
                                 ds->label,
                                 time_attr,
                                 iter_attr,
                                 data,
                                 n1,
                                 n2,
                                 axis1,
                                 axis2,
                                 &fixed);
}

/* ------------------------------ dg_run_1d --------------------------------- */

int dg_run_1d(const DG_Request1D *req,
              const DG_DatasetDesc *datasets,
              size_t ndatasets,
              DG_EvalFn eval_fn,
              void *eval_ctx,
              const DG_RunOptions *opt_in,
              const char *path_prefix,
              MPI_Comm comm)
{
    if (!req || !datasets || ndatasets == 0 || !eval_fn || !path_prefix) return 1;
    const int vrc = dg_validate_req_1d(req);
    if (vrc) return 100 + vrc;

    DG_RunOptions opt = opt_in ? *opt_in : dg_default_run_options();

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = opt.root_rank;

    size_t n1 = 0;
    double *a1 = materialize_axis_values(&req->a1, &n1);
    if (!a1) return 2;

    const DiagAxis axis1 = make_diag_axis(&req->a1, a1, n1);

    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    if (!counts || !displs)
    {
        free(a1);
        free(counts);
        free(displs);
        return 3;
    }
    compute_counts_displs(size, (int)n1, counts, displs);

    const int local_n = counts[rank];
    const int local_0 = displs[rank];

    /* Local buffer is point-major: [local_n][ndatasets] */
    float *local = (float *)malloc((size_t)local_n * ndatasets * sizeof(float));
    if (!local)
    {
        free(a1); free(counts); free(displs);
        return 4;
    }

    float *tmp = (float *)malloc(ndatasets * sizeof(float));
    if (!tmp)
    {
        free(local);
        free(a1); free(counts); free(displs);
        return 5;
    }

    for (int il = 0; il < local_n; ++il)
    {
        const size_t i = (size_t)(local_0 + il);

        DG_FixedCoords c = req->fixed;
        apply_axis(req->a1.id, a1[i], &c);

        int rc = eval_fn(&c, eval_ctx, tmp, ndatasets);
        if (rc != 0)
        {
            for (size_t d = 0; d < ndatasets; ++d) tmp[d] = (float)NAN;
        }

        for (size_t d = 0; d < ndatasets; ++d)
            local[(size_t)il * ndatasets + d] = tmp[d];
    }

    free(tmp);

    /* Root global buffer for one dataset at a time (matches current H5 writers). */
    float *global = NULL;
    if (rank == root)
    {
        global = (float *)malloc(n1 * sizeof(float));
        if (!global)
        {
            free(local);
            free(a1); free(counts); free(displs);
            return 6;
        }
    }

    int write_rc = 0;

    /* Gather each dataset separately to avoid packing multi-dataset H5. */
    float *local_d = (float *)malloc((size_t)local_n * sizeof(float));
    if (!local_d)
    {
        if (rank == root) free(global);
        free(local);
        free(a1); free(counts); free(displs);
        return 7;
    }

    for (size_t d = 0; d < ndatasets; ++d)
    {
        for (int il = 0; il < local_n; ++il)
            local_d[il] = local[(size_t)il * ndatasets + d];

        MPI_Gatherv(local_d, local_n, MPI_FLOAT,
                    global, counts, displs, MPI_FLOAT,
                    root, comm);

        if (rank == root)
        {
            const int rc = write_1d_one(req, &datasets[d], global, n1, &axis1,
                                       opt.write_time_iter_0, path_prefix);
            if (rc != 0 && write_rc == 0) write_rc = rc;
        }
    }

    free(local_d);
    if (rank == root) free(global);

    free(local);
    free(a1);
    free(counts);
    free(displs);

    return write_rc;
}

/* ------------------------------ dg_run_2d --------------------------------- */

int dg_run_2d(const DG_Request2D *req,
              const DG_DatasetDesc *datasets,
              size_t ndatasets,
              DG_EvalFn eval_fn,
              void *eval_ctx,
              const DG_RunOptions *opt_in,
              const char *path_prefix,
              MPI_Comm comm)
{
    if (!req || !datasets || ndatasets == 0 || !eval_fn || !path_prefix) return 1;
    const int vrc = dg_validate_req_2d(req);
    if (vrc) return 100 + vrc;

    DG_RunOptions opt = opt_in ? *opt_in : dg_default_run_options();

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = opt.root_rank;

    size_t n1 = 0, n2 = 0;
    double *a1 = materialize_axis_values(&req->a1, &n1);
    double *a2 = materialize_axis_values(&req->a2, &n2);
    if (!a1 || !a2)
    {
        free(a1); free(a2);
        return 2;
    }

    const DiagAxis axis1 = make_diag_axis(&req->a1, a1, n1);
    const DiagAxis axis2 = make_diag_axis(&req->a2, a2, n2);

    const size_t ngrid = n1 * n2;

    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    if (!counts || !displs)
    {
        free(a1); free(a2);
        free(counts); free(displs);
        return 3;
    }
    compute_counts_displs(size, (int)ngrid, counts, displs);

    const int local_n = counts[rank];
    const int local_0 = displs[rank];

    float *local = (float *)malloc((size_t)local_n * ndatasets * sizeof(float));
    if (!local)
    {
        free(a1); free(a2); free(counts); free(displs);
        return 4;
    }

    float *tmp = (float *)malloc(ndatasets * sizeof(float));
    if (!tmp)
    {
        free(local);
        free(a1); free(a2); free(counts); free(displs);
        return 5;
    }

    for (int il = 0; il < local_n; ++il)
    {
        const size_t g = (size_t)(local_0 + il);
        const size_t i2 = g / n1;
        const size_t i1 = g - i2 * n1;

        DG_FixedCoords c = req->fixed;
        apply_axis(req->a1.id, a1[i1], &c);
        apply_axis(req->a2.id, a2[i2], &c);

        int rc = eval_fn(&c, eval_ctx, tmp, ndatasets);
        if (rc != 0)
        {
            for (size_t d = 0; d < ndatasets; ++d) tmp[d] = (float)NAN;
        }

        for (size_t d = 0; d < ndatasets; ++d)
            local[(size_t)il * ndatasets + d] = tmp[d];
    }

    free(tmp);

    float *global = NULL;
    if (rank == root)
    {
        global = (float *)malloc(ngrid * sizeof(float));
        if (!global)
        {
            free(local);
            free(a1); free(a2); free(counts); free(displs);
            return 6;
        }
    }

    float *local_d = (float *)malloc((size_t)local_n * sizeof(float));
    if (!local_d)
    {
        if (rank == root) free(global);
        free(local);
        free(a1); free(a2); free(counts); free(displs);
        return 7;
    }

    int write_rc = 0;

    for (size_t d = 0; d < ndatasets; ++d)
    {
        for (int il = 0; il < local_n; ++il)
            local_d[il] = local[(size_t)il * ndatasets + d];

        MPI_Gatherv(local_d, local_n, MPI_FLOAT,
                    global, counts, displs, MPI_FLOAT,
                    root, comm);

        if (rank == root)
        {
            const int rc = write_2d_one(req, &datasets[d], global, n1, n2, &axis1, &axis2,
                                       opt.write_time_iter_0, path_prefix);
            if (rc != 0 && write_rc == 0) write_rc = rc;
        }
    }

    free(local_d);
    if (rank == root) free(global);

    free(local);
    free(a1);
    free(a2);
    free(counts);
    free(displs);

    return write_rc;
}

