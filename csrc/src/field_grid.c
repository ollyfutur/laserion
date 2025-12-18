#include "field_grid.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ----------------------------- small helpers ----------------------------- */

static int is_valid_axis(FG_Axis id)
{
    return (id == FG_T || id == FG_X || id == FG_Y || id == FG_Z);
}

int fg_validate_axis(const FG_AxisSpec *a)
{
    if (!a)
        return 1;
    if (!is_valid_axis(a->id))
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
    if (opt->quantity_mask == 0)
        return 2;
    if (opt->component_mask == 0)
        return 3;

    if (opt->quantity_mask & ~((uint32_t)FG_Q_E | (uint32_t)FG_Q_A))
        return 4;
    if (opt->component_mask & ~((uint32_t)FG_C_ALL))
        return 5;

    if (!(opt->write_mode == FG_WRITE_SINGLE_FILE || opt->write_mode == FG_WRITE_SPLIT_FILES))
        return 6;
    return 0;
}

int fg_validate_request_1d(const FG_Request1D *r, const FG_Options *opt)
{
    if (!r || !opt)
        return 1;
    int rc = fg_validate_options(opt);
    if (rc)
        return 2;
    rc = fg_validate_axis(&r->a1);
    if (rc)
        return 3;
    return 0;
}

int fg_validate_request_2d(const FG_Request2D *r, const FG_Options *opt)
{
    if (!r || !opt)
        return 1;
    int rc = fg_validate_options(opt);
    if (rc)
        return 2;
    rc = fg_validate_axis(&r->a1);
    if (rc)
        return 3;
    rc = fg_validate_axis(&r->a2);
    if (rc)
        return 4;
    if (r->a1.id == r->a2.id)
        return 5;
    return 0;
}

static double *materialize_axis_values(const FG_AxisSpec *a, size_t *n_out)
{
    *n_out = 0;
    if (a->kind == FG_AXIS_VALUES)
    {
        const size_t n = a->n_values;
        double *v = (double *)malloc(sizeof(double) * n);
        if (!v)
            return NULL;
        memcpy(v, a->values, sizeof(double) * n);
        *n_out = n;
        return v;
    }
    else
    {
        const size_t n = a->n;
        double *v = (double *)malloc(sizeof(double) * n);
        if (!v)
            return NULL;

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
}

static void axis_apply(FG_Axis id, double val, double *t_fs, double r_um[3])
{
    switch (id)
    {
    case FG_T:
        *t_fs = val;
        break;
    case FG_X:
        r_um[0] = val;
        break;
    case FG_Y:
        r_um[1] = val;
        break;
    case FG_Z:
        r_um[2] = val;
        break;
    default:
        break;
    }
}

static void compute_counts_displs(int size, int n, int *counts, int *displs)
{
    const int base = (size > 0) ? (n / size) : 0;
    const int rem = (size > 0) ? (n % size) : 0;

    int disp = 0;
    for (int r = 0; r < size; ++r)
    {
        const int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt;
        displs[r] = disp;
        disp += cnt;
    }
}

static size_t count_datasets(uint32_t qmask, uint32_t cmask)
{
    size_t n = 0;
    if (qmask & (uint32_t)FG_Q_E)
    {
        if (cmask & (uint32_t)FG_C_X)
            ++n;
        if (cmask & (uint32_t)FG_C_Y)
            ++n;
        if (cmask & (uint32_t)FG_C_Z)
            ++n;
    }
    if (qmask & (uint32_t)FG_Q_A)
    {
        if (cmask & (uint32_t)FG_C_X)
            ++n;
        if (cmask & (uint32_t)FG_C_Y)
            ++n;
        if (cmask & (uint32_t)FG_C_Z)
            ++n;
    }
    return n;
}

static void fill_datasets(uint32_t qmask, uint32_t cmask,
                          const char **names,
                          const char **units,
                          const char **long_names)
{
    size_t k = 0;

    if (qmask & (uint32_t)FG_Q_E)
    {
        if (cmask & (uint32_t)FG_C_X)
        {
            names[k] = "Ex";
            units[k] = "GV/m";
            long_names[k] = "E_x";
            ++k;
        }
        if (cmask & (uint32_t)FG_C_Y)
        {
            names[k] = "Ey";
            units[k] = "GV/m";
            long_names[k] = "E_y";
            ++k;
        }
        if (cmask & (uint32_t)FG_C_Z)
        {
            names[k] = "Ez";
            units[k] = "GV/m";
            long_names[k] = "E_z";
            ++k;
        }
    }
    if (qmask & (uint32_t)FG_Q_A)
    {
        if (cmask & (uint32_t)FG_C_X)
        {
            names[k] = "Ax";
            units[k] = "a.u.";
            long_names[k] = "A_x";
            ++k;
        }
        if (cmask & (uint32_t)FG_C_Y)
        {
            names[k] = "Ay";
            units[k] = "a.u.";
            long_names[k] = "A_y";
            ++k;
        }
        if (cmask & (uint32_t)FG_C_Z)
        {
            names[k] = "Az";
            units[k] = "a.u.";
            long_names[k] = "A_z";
            ++k;
        }
    }
}

static DiagAxis make_diag_axis(const FG_AxisSpec *spec, const double *vals, size_t n)
{
    DiagAxis a = (DiagAxis){0};
    a.id = (spec->id == FG_T ? DIAG_AXIS_T : (spec->id == FG_X ? DIAG_AXIS_X : (spec->id == FG_Y ? DIAG_AXIS_Y : DIAG_AXIS_Z)));
    a.long_name = spec->name;
    a.units = spec->units;
    a.vmin = (n > 0) ? vals[0] : 0.0;
    a.vmax = (n > 0) ? vals[n - 1] : 0.0;
    return a;
}

/* Write one dataset either into a single shared file or as split file. */
static int write_dataset_1d(const FG_Options *opt,
                            const char *path_or_prefix,
                            const char *dset_name, const char *dset_units, const char *label,
                            const float *data, size_t n1,
                            const DiagAxis *axis1,
                            const DiagFixedCoords *fixed)
{
    const double time = 0.0;
    const int iter = 0;

    if (opt->write_mode == FG_WRITE_SINGLE_FILE)
    {
        return diag_h5_write_grid_1d(path_or_prefix,
                                     dset_name, dset_units, label,
                                     time, iter,
                                     data, n1,
                                     axis1, fixed);
    }
    else
    {
        char path[512];
        snprintf(path, sizeof(path), "%s_%s.h5", path_or_prefix, dset_name);
        return diag_h5_write_grid_1d(path,
                                     dset_name, dset_units, label,
                                     time, iter,
                                     data, n1,
                                     axis1, fixed);
    }
}

static int write_dataset_2d(const FG_Options *opt,
                            const char *path_or_prefix,
                            const char *dset_name, const char *dset_units, const char *label,
                            const float *data, size_t n1, size_t n2,
                            const DiagAxis *axis1, const DiagAxis *axis2,
                            const DiagFixedCoords *fixed)
{
    const double time = 0.0;
    const int iter = 0;

    if (opt->write_mode == FG_WRITE_SINGLE_FILE)
    {
        return diag_h5_write_grid_2d(path_or_prefix,
                                     dset_name, dset_units, label,
                                     time, iter,
                                     data, n1, n2,
                                     axis1, axis2, fixed);
    }
    else
    {
        char path[512];
        snprintf(path, sizeof(path), "%s_%s.h5", path_or_prefix, dset_name);
        return diag_h5_write_grid_2d(path,
                                     dset_name, dset_units, label,
                                     time, iter,
                                     data, n1, n2,
                                     axis1, axis2,
                                     fixed);
    }
}

/* ------------------------------ core kernels ------------------------------ */

static int run_generic_1d(const struct LaserPulse *pulse,
                          const FG_Request1D *req,
                          const FG_Options *opt,
                          const char *path_or_prefix,
                          MPI_Comm comm)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = opt->root_rank;

    size_t n1 = 0;
    double *a1 = materialize_axis_values(&req->a1, &n1);
    if (!a1)
        return 10;

    const uint32_t qmask = opt->quantity_mask;
    const uint32_t cmask = opt->component_mask;
    const size_t nds = count_datasets(qmask, cmask);

    const char **names = (const char **)calloc(nds, sizeof(char *));
    const char **units = (const char **)calloc(nds, sizeof(char *));
    const char **longn = (const char **)calloc(nds, sizeof(char *));
    if (!names || !units || !longn)
    {
        free(a1);
        free(names);
        free(units);
        free(longn);
        return 11;
    }
    fill_datasets(qmask, cmask, names, units, longn);

    /* Decompose along the 1D index */
    int *counts = (int *)calloc((size_t)size, sizeof(int));
    int *displs = (int *)calloc((size_t)size, sizeof(int));
    if (!counts || !displs)
    {
        free(a1);
        free(names);
        free(units);
        free(longn);
        free(counts);
        free(displs);
        return 12;
    }
    compute_counts_displs(size, (int)n1, counts, displs);

    const int local_n = counts[rank];
    const int local_0 = displs[rank];

    /* local buffers: nds arrays of length local_n */
    float **local = (float **)calloc(nds, sizeof(float *));
    if (!local)
    {
        free(a1);
        free(names);
        free(units);
        free(longn);
        free(counts);
        free(displs);
        return 13;
    }
    for (size_t k = 0; k < nds; ++k)
    {
        local[k] = (float *)malloc(sizeof(float) * (size_t)local_n);
        if (!local[k])
        {
            for (size_t j = 0; j < k; ++j)
                free(local[j]);
            free(local);
            free(a1);
            free(names);
            free(units);
            free(longn);
            free(counts);
            free(displs);
            return 14;
        }
    }

    /* Evaluate */
    for (int il = 0; il < local_n; ++il)
    {
        const size_t i = (size_t)(local_0 + il);

        double t_fs = req->t0_fs;
        double r_um[3] = {req->x0_um, req->y0_um, req->z0_um};
        axis_apply(req->a1.id, a1[i], &t_fs, r_um);

        double E[3] = {0, 0, 0};
        double A[3] = {0, 0, 0};

        if (qmask & (uint32_t)FG_Q_E)
            LaserPulse_E(pulse, t_fs, r_um, E);
        if (qmask & (uint32_t)FG_Q_A)
            LaserPulse_A(pulse, t_fs, r_um, A);

        size_t k = 0;
        if (qmask & (uint32_t)FG_Q_E)
        {
            if (cmask & (uint32_t)FG_C_X)
                local[k++][il] = (float)E[0];
            if (cmask & (uint32_t)FG_C_Y)
                local[k++][il] = (float)E[1];
            if (cmask & (uint32_t)FG_C_Z)
                local[k++][il] = (float)E[2];
        }
        if (qmask & (uint32_t)FG_Q_A)
        {
            if (cmask & (uint32_t)FG_C_X)
                local[k++][il] = (float)A[0];
            if (cmask & (uint32_t)FG_C_Y)
                local[k++][il] = (float)A[1];
            if (cmask & (uint32_t)FG_C_Z)
                local[k++][il] = (float)A[2];
        }
    }

    /* Root receives full buffers and writes */
    float **full = NULL;
    if (rank == root)
    {
        full = (float **)calloc(nds, sizeof(float *));
        if (!full)
        {
            for (size_t k = 0; k < nds; ++k)
                free(local[k]);
            free(local);
            free(a1);
            free(names);
            free(units);
            free(longn);
            free(counts);
            free(displs);
            return 15;
        }
        for (size_t k = 0; k < nds; ++k)
        {
            full[k] = (float *)malloc(sizeof(float) * n1);
            if (!full[k])
            {
                for (size_t j = 0; j < k; ++j)
                    free(full[j]);
                free(full);
                for (size_t kk = 0; kk < nds; ++kk)
                    free(local[kk]);
                free(local);
                free(a1);
                free(names);
                free(units);
                free(longn);
                free(counts);
                free(displs);
                return 16;
            }
        }
    }

    int gather_err = 0;
    for (size_t k = 0; k < nds; ++k)
    {
        int rc = MPI_Gatherv(local[k], local_n, MPI_FLOAT,
                             (rank == root) ? full[k] : NULL,
                             counts, displs, MPI_FLOAT,
                             root, comm);
        if (rc != MPI_SUCCESS)
            gather_err = 1;
    }
    int gather_err_all = 0;
    MPI_Allreduce(&gather_err, &gather_err_all, 1, MPI_INT, MPI_MAX, comm);

    /* free locals */
    for (size_t k = 0; k < nds; ++k)
        free(local[k]);
    free(local);
    free(counts);
    free(displs);

    if (gather_err_all)
    {
        if (rank == root)
        {
            for (size_t k = 0; k < nds; ++k)
                free(full[k]);
            free(full);
        }
        free(a1);
        free(names);
        free(units);
        free(longn);
        return 17;
    }

    if (rank == root)
    {
        DiagAxis axis1 = make_diag_axis(&req->a1, a1, n1);

        DiagFixedCoords fixed = (DiagFixedCoords){
            .t = req->t0_fs,
            .x = req->x0_um,
            .y = req->y0_um,
            .z = req->z0_um};

        for (size_t k = 0; k < nds; ++k)
        {
            int rc = write_dataset_1d(opt, path_or_prefix,
                                      names[k], units[k], longn[k],
                                      full[k], n1,
                                      &axis1, &fixed);
            if (rc != 0)
            {
                for (size_t j = 0; j < nds; ++j)
                    free(full[j]);
                free(full);
                free(a1);
                free(names);
                free(units);
                free(longn);
                return 18;
            }
        }

        for (size_t k = 0; k < nds; ++k)
            free(full[k]);
        free(full);
    }

    free(a1);
    free(names);
    free(units);
    free(longn);
    return 0;
}

static int run_generic_2d(const struct LaserPulse *pulse,
                          const FG_Request2D *req,
                          const FG_Options *opt,
                          const char *path_or_prefix,
                          MPI_Comm comm)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = opt->root_rank;

    size_t n1 = 0, n2 = 0;
    double *a1 = materialize_axis_values(&req->a1, &n1);
    double *a2 = materialize_axis_values(&req->a2, &n2);
    if (!a1 || !a2)
    {
        free(a1);
        free(a2);
        return 20;
    }

    const uint32_t qmask = opt->quantity_mask;
    const uint32_t cmask = opt->component_mask;
    const size_t nds = count_datasets(qmask, cmask);

    const char **names = (const char **)calloc(nds, sizeof(char *));
    const char **units = (const char **)calloc(nds, sizeof(char *));
    const char **longn = (const char **)calloc(nds, sizeof(char *));
    if (!names || !units || !longn)
    {
        free(a1);
        free(a2);
        free(names);
        free(units);
        free(longn);
        return 21;
    }
    fill_datasets(qmask, cmask, names, units, longn);

    /* Flattened grid index: i = i1 + n1*i2  (axis1 fastest) */
    const size_t ngrid = n1 * n2;

    int *counts = (int *)calloc((size_t)size, sizeof(int));
    int *displs = (int *)calloc((size_t)size, sizeof(int));
    if (!counts || !displs)
    {
        free(a1);
        free(a2);
        free(names);
        free(units);
        free(longn);
        free(counts);
        free(displs);
        return 22;
    }
    compute_counts_displs(size, (int)ngrid, counts, displs);

    const int local_n = counts[rank];
    const int local_0 = displs[rank];

    float **local = (float **)calloc(nds, sizeof(float *));
    if (!local)
    {
        free(a1);
        free(a2);
        free(names);
        free(units);
        free(longn);
        free(counts);
        free(displs);
        return 23;
    }
    for (size_t k = 0; k < nds; ++k)
    {
        local[k] = (float *)malloc(sizeof(float) * (size_t)local_n);
        if (!local[k])
        {
            for (size_t j = 0; j < k; ++j)
                free(local[j]);
            free(local);
            free(a1);
            free(a2);
            free(names);
            free(units);
            free(longn);
            free(counts);
            free(displs);
            return 24;
        }
    }

    for (int il = 0; il < local_n; ++il)
    {
        const size_t ig = (size_t)(local_0 + il);
        const size_t i1 = ig % n1;
        const size_t i2 = ig / n1;

        double t_fs = req->t0_fs;
        double r_um[3] = {req->x0_um, req->y0_um, req->z0_um};

        axis_apply(req->a1.id, a1[i1], &t_fs, r_um);
        axis_apply(req->a2.id, a2[i2], &t_fs, r_um);

        double E[3] = {0, 0, 0};
        double A[3] = {0, 0, 0};

        if (qmask & (uint32_t)FG_Q_E)
            LaserPulse_E(pulse, t_fs, r_um, E);
        if (qmask & (uint32_t)FG_Q_A)
            LaserPulse_A(pulse, t_fs, r_um, A);

        size_t k = 0;
        if (qmask & (uint32_t)FG_Q_E)
        {
            if (cmask & (uint32_t)FG_C_X)
                local[k++][il] = (float)E[0];
            if (cmask & (uint32_t)FG_C_Y)
                local[k++][il] = (float)E[1];
            if (cmask & (uint32_t)FG_C_Z)
                local[k++][il] = (float)E[2];
        }
        if (qmask & (uint32_t)FG_Q_A)
        {
            if (cmask & (uint32_t)FG_C_X)
                local[k++][il] = (float)A[0];
            if (cmask & (uint32_t)FG_C_Y)
                local[k++][il] = (float)A[1];
            if (cmask & (uint32_t)FG_C_Z)
                local[k++][il] = (float)A[2];
        }
    }

    float **full = NULL;
    if (rank == root)
    {
        full = (float **)calloc(nds, sizeof(float *));
        if (!full)
        {
            for (size_t k = 0; k < nds; ++k)
                free(local[k]);
            free(local);
            free(a1);
            free(a2);
            free(names);
            free(units);
            free(longn);
            free(counts);
            free(displs);
            return 25;
        }
        for (size_t k = 0; k < nds; ++k)
        {
            full[k] = (float *)malloc(sizeof(float) * ngrid);
            if (!full[k])
            {
                for (size_t j = 0; j < k; ++j)
                    free(full[j]);
                free(full);
                for (size_t kk = 0; kk < nds; ++kk)
                    free(local[kk]);
                free(local);
                free(a1);
                free(a2);
                free(names);
                free(units);
                free(longn);
                free(counts);
                free(displs);
                return 26;
            }
        }
    }

    int gather_err = 0;
    for (size_t k = 0; k < nds; ++k)
    {
        int rc = MPI_Gatherv(local[k], local_n, MPI_FLOAT,
                             (rank == root) ? full[k] : NULL,
                             counts, displs, MPI_FLOAT,
                             root, comm);
        if (rc != MPI_SUCCESS)
            gather_err = 1;
    }
    int gather_err_all = 0;
    MPI_Allreduce(&gather_err, &gather_err_all, 1, MPI_INT, MPI_MAX, comm);

    for (size_t k = 0; k < nds; ++k)
        free(local[k]);
    free(local);
    free(counts);
    free(displs);

    if (gather_err_all)
    {
        if (rank == root)
        {
            for (size_t k = 0; k < nds; ++k)
                free(full[k]);
            free(full);
        }
        free(a1);
        free(a2);
        free(names);
        free(units);
        free(longn);
        return 27;
    }

    if (rank == root)
    {
        DiagAxis axis1 = make_diag_axis(&req->a1, a1, n1);
        DiagAxis axis2 = make_diag_axis(&req->a2, a2, n2);

        DiagFixedCoords fixed = (DiagFixedCoords){
            .t = req->t0_fs,
            .x = req->x0_um,
            .y = req->y0_um,
            .z = req->z0_um};

        for (size_t k = 0; k < nds; ++k)
        {
            int rc = write_dataset_2d(opt, path_or_prefix,
                                      names[k], units[k], longn[k],
                                      full[k], n1, n2,
                                      &axis1, &axis2, &fixed);
            if (rc != 0)
            {
                for (size_t j = 0; j < nds; ++j)
                    free(full[j]);
                free(full);
                free(a1);
                free(a2);
                free(names);
                free(units);
                free(longn);
                return 28;
            }
        }

        for (size_t k = 0; k < nds; ++k)
            free(full[k]);
        free(full);
    }

    free(a1);
    free(a2);
    free(names);
    free(units);
    free(longn);
    return 0;
}

/* ------------------------------ public API -------------------------------- */

int fg_run_1d(const struct LaserPulse *pulse,
              const FG_Request1D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm)
{
    if (!pulse || !req || !opt || !path_or_prefix)
        return 1;
    if (fg_validate_request_1d(req, opt) != 0)
        return 2;
    return run_generic_1d(pulse, req, opt, path_or_prefix, comm);
}

int fg_run_2d(const struct LaserPulse *pulse,
              const FG_Request2D *req,
              const FG_Options *opt,
              const char *path_or_prefix,
              MPI_Comm comm)
{
    if (!pulse || !req || !opt || !path_or_prefix)
        return 1;
    if (fg_validate_request_2d(req, opt) != 0)
        return 2;
    return run_generic_2d(pulse, req, opt, path_or_prefix, comm);
}
