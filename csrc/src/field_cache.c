#include "field_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdarg.h>
#include <hdf5.h>
#include <unistd.h>

static void status_root(MPI_Comm comm, int root, const char *msg)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        printf("%s\n", msg);
        fflush(stdout);
    }
}

static void status_rootf(MPI_Comm comm, int root, const char *fmt, ...)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != root)
        return;

    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static int ensure_parent_dir(const char *prefix)
{
    // prefix example: "out/fields"
    // We want to mkdir -p "out" (and any parents if you use "a/b/c/fields")

    char tmp[512];
    size_t n = strlen(prefix);
    if (n >= sizeof(tmp))
        return 1;

    strcpy(tmp, prefix);

    // Find last '/'
    char *slash = strrchr(tmp, '/');
    if (!slash)
        return 0; // no directory component, current dir is fine

    *slash = '\0'; // tmp now holds the directory path

    // mkdir -p implementation
    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p == '/')
        {
            *p = '\0';
            (void)mkdir(tmp, 0777);
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0)
    {
        // If already exists, fine
        struct stat st;
        if (stat(tmp, &st) != 0)
            return 2;
        if (!S_ISDIR(st.st_mode))
            return 3;
    }
    return 0;
}

static void die_root(MPI_Comm comm, int root, const char *msg)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
        fprintf(stderr, "%s\n", msg);
}

FieldCacheOptions field_cache_default_options(void)
{
    FieldCacheOptions o;
    o.compute_A = 1;
    o.merge_on_root = 1;
    o.root_rank = 0;
    return o;
}

static const char *axis_name(Axis a)
{
    switch (a)
    {
    case AXIS_X:
        return "x";
    case AXIS_Y:
        return "y";
    case AXIS_Z:
        return "z";
    default:
        return "?";
    }
}

static void set_r_from_axes(const InputGridSpec *g,
                            double a1, double a2,
                            double r_um[3])
{
    r_um[0] = g->fixed_x;
    r_um[1] = g->fixed_y;
    r_um[2] = g->fixed_z;

    switch (g->ax1)
    {
    case AXIS_X:
        r_um[0] = a1;
        break;
    case AXIS_Y:
        r_um[1] = a1;
        break;
    case AXIS_Z:
        r_um[2] = a1;
        break;
    }

    if (g->has_ax2)
    {
        switch (g->ax2)
        {
        case AXIS_X:
            r_um[0] = a2;
            break;
        case AXIS_Y:
            r_um[1] = a2;
            break;
        case AXIS_Z:
            r_um[2] = a2;
            break;
        }
    }
}

static void decompose_1d(size_t n, int rank, int nranks, size_t *i0, size_t *nloc)
{
    size_t base = n / (size_t)nranks;
    size_t rem = n % (size_t)nranks;

    size_t start = (size_t)rank * base + (size_t)((rank < (int)rem) ? rank : rem);
    size_t count = base + (size_t)((rank < (int)rem) ? 1 : 0);

    *i0 = start;
    *nloc = count;
}

static hid_t h5_create_or_fail(const char *path)
{
    return H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

static hid_t h5_open_ro_or_fail(const char *path)
{
    return H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
}

static int h5_write_attr_string(hid_t obj, const char *name, const char *value)
{
    hid_t atype = H5Tcopy(H5T_C_S1);
    if (atype < 0)
        return 1;
    H5Tset_size(atype, strlen(value));
    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
    {
        H5Tclose(atype);
        return 2;
    }

    hid_t attr = H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        H5Tclose(atype);
        return 3;
    }

    herr_t st = H5Awrite(attr, atype, value);
    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);
    return (st < 0) ? 4 : 0;
}

static int h5_write_attr_double(hid_t obj, const char *name, double v)
{
    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
        return 1;
    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_DOUBLE, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        return 2;
    }
    herr_t st = H5Awrite(attr, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(attr);
    H5Sclose(aspace);
    return (st < 0) ? 3 : 0;
}

static int h5_write_attr_int(hid_t obj, const char *name, int v)
{
    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
        return 1;
    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_INT, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        return 2;
    }
    herr_t st = H5Awrite(attr, H5T_NATIVE_INT, &v);
    H5Aclose(attr);
    H5Sclose(aspace);
    return (st < 0) ? 3 : 0;
}

static int create_single_dataset_file(hid_t *out_f, hid_t *out_dset,
                                      const char *path,
                                      const char *component_name,
                                      int nd, const hsize_t *dims,
                                      const InputGridSpec *g,
                                      int is_rank_file,
                                      int ax_i0, int ax_nloc)
{
    hid_t f = h5_create_or_fail(path);
    if (f < 0)
        return 1;

    // metadata
    h5_write_attr_string(f, "TYPE", "grid");
    h5_write_attr_string(f, "layout", "field_cache");
    h5_write_attr_string(f, "component", component_name);
    h5_write_attr_int(f, "ndim_total", nd);

    h5_write_attr_string(f, "axis1", "t");
    h5_write_attr_double(f, "t_min_fs", g->t_min);
    h5_write_attr_double(f, "t_max_fs", g->t_max);
    h5_write_attr_double(f, "dt_fs", g->dt);
    h5_write_attr_int(f, "t_n", g->t_n);

    h5_write_attr_string(f, "axis2", axis_name(g->ax1));
    h5_write_attr_double(f, "ax1_min_um", g->ax1_min);
    h5_write_attr_double(f, "ax1_max_um", g->ax1_max);
    h5_write_attr_double(f, "dx1_um", g->dx1);
    h5_write_attr_int(f, "ax1_n", g->ax1_n);

    if (nd == 3)
    {
        h5_write_attr_string(f, "axis3", axis_name(g->ax2));
        h5_write_attr_double(f, "ax2_min_um", g->ax2_min);
        h5_write_attr_double(f, "ax2_max_um", g->ax2_max);
        h5_write_attr_double(f, "dx2_um", g->dx2);
        h5_write_attr_int(f, "ax2_n", g->ax2_n);
    }

    h5_write_attr_double(f, "fixed_x_um", g->fixed_x);
    h5_write_attr_double(f, "fixed_y_um", g->fixed_y);
    h5_write_attr_double(f, "fixed_z_um", g->fixed_z);

    if (is_rank_file)
    {
        h5_write_attr_int(f, "slab_i0", ax_i0);
        h5_write_attr_int(f, "slab_nloc", ax_nloc);
    }

    // single dataset
    hid_t space = H5Screate_simple(nd, dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 2;
    }

    hid_t dset = H5Dcreate2(f, "/data", H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);
    if (dset < 0)
    {
        H5Fclose(f);
        return 3;
    }

    *out_f = f;
    *out_dset = dset;
    return 0;
}

static void component_filename(char *buf, size_t bufsz,
                               const char *prefix,
                               const char *comp, int rank, int is_rank)
{
    if (is_rank)
    {
        snprintf(buf, bufsz, "%s_%s.rank%04d.h5", prefix, comp, rank);
    }
    else
    {
        snprintf(buf, bufsz, "%s_%s.h5", prefix, comp);
    }
}

static void remove_rank_files(const char *prefix, const char *comp, int nranks, int root, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != root)
        return;

    char path[512];

    for (int r = 0; r < nranks; ++r)
    {
        // same naming function you already have:
        component_filename(path, sizeof(path), prefix, comp, r, 1);
        if (unlink(path) != 0)
        {
            // Best-effort: warn, do not abort
            fprintf(stderr, "field_cache: warning: could not remove partial file \"%s\"\n", path);
        }
    }
}

static int write_rank_files_2d(const InputGridSpec *g,
                               const LaserPulse *pulse,
                               const char *prefix,
                               int compute_A,
                               size_t a1_i0, size_t a1_nloc,
                               int rank)
{
    // local dims: [t, a1_local]
    hsize_t dims[2] = {(hsize_t)g->t_n, (hsize_t)a1_nloc};

    // open one file per component
    hid_t fEx, dEx, fEy, dEy, fEz, dEz;
    char path[512];

    component_filename(path, sizeof(path), prefix, "Ex", rank, 1);
    if (create_single_dataset_file(&fEx, &dEx, path, "Ex", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc) != 0)
        return 1;
    component_filename(path, sizeof(path), prefix, "Ey", rank, 1);
    if (create_single_dataset_file(&fEy, &dEy, path, "Ey", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc) != 0)
        return 2;
    component_filename(path, sizeof(path), prefix, "Ez", rank, 1);
    if (create_single_dataset_file(&fEz, &dEz, path, "Ez", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc) != 0)
        return 3;

    hid_t fAx = -1, dAx = -1, fAy = -1, dAy = -1, fAz = -1, dAz = -1;
    if (compute_A)
    {
        component_filename(path, sizeof(path), prefix, "Ax", rank, 1);
        if (create_single_dataset_file(&fAx, &dAx, path, "Ax", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc) != 0)
            return 4;
        component_filename(path, sizeof(path), prefix, "Ay", rank, 1);
        if (create_single_dataset_file(&fAy, &dAy, path, "Ay", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc) != 0)
            return 5;
        component_filename(path, sizeof(path), prefix, "Az", rank, 1);
        if (create_single_dataset_file(&fAz, &dAz, path, "Az", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc) != 0)
            return 6;
    }

    // streaming buffers: one row (a1_local) per time, per component
    double *bx = (double *)calloc(a1_nloc, sizeof(double));
    double *by = (double *)calloc(a1_nloc, sizeof(double));
    double *bz = (double *)calloc(a1_nloc, sizeof(double));
    double *bax = compute_A ? (double *)calloc(a1_nloc, sizeof(double)) : NULL;
    double *bay = compute_A ? (double *)calloc(a1_nloc, sizeof(double)) : NULL;
    double *baz = compute_A ? (double *)calloc(a1_nloc, sizeof(double)) : NULL;

    if (!bx || !by || !bz || (compute_A && (!bax || !bay || !baz)))
        return 7;

    for (int it = 0; it < g->t_n; ++it)
    {
        double t = g->t_min + (double)it * g->dt;

        for (size_t ia = 0; ia < a1_nloc; ++ia)
        {
            size_t ig = a1_i0 + ia;
            double a1 = g->ax1_min + (double)ig * g->dx1;

            double r[3];
            set_r_from_axes(g, a1, 0.0, r);

            double E[3];
            LaserPulse_E(pulse, t, r, E);
            bx[ia] = E[0];
            by[ia] = E[1];
            bz[ia] = E[2];

            if (compute_A)
            {
                double A[3];
                LaserPulse_A(pulse, t, r, A);
                bax[ia] = A[0];
                bay[ia] = A[1];
                baz[ia] = A[2];
            }
        }

        // hyperslab [it, :]
        hsize_t start[2] = {(hsize_t)it, 0};
        hsize_t count[2] = {1, (hsize_t)a1_nloc};
        hid_t mspace = H5Screate_simple(2, count, NULL);

        hid_t fspace = H5Dget_space(dEx);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Dwrite(dEx, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, bx);
        H5Sclose(fspace);

        fspace = H5Dget_space(dEy);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Dwrite(dEy, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, by);
        H5Sclose(fspace);

        fspace = H5Dget_space(dEz);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Dwrite(dEz, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, bz);
        H5Sclose(fspace);

        if (compute_A)
        {
            fspace = H5Dget_space(dAx);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
            H5Dwrite(dAx, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, bax);
            H5Sclose(fspace);

            fspace = H5Dget_space(dAy);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
            H5Dwrite(dAy, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, bay);
            H5Sclose(fspace);

            fspace = H5Dget_space(dAz);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
            H5Dwrite(dAz, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, baz);
            H5Sclose(fspace);
        }

        H5Sclose(mspace);
    }

    free(bx);
    free(by);
    free(bz);
    free(bax);
    free(bay);
    free(baz);

    H5Dclose(dEx);
    H5Fclose(fEx);
    H5Dclose(dEy);
    H5Fclose(fEy);
    H5Dclose(dEz);
    H5Fclose(fEz);

    if (compute_A)
    {
        H5Dclose(dAx);
        H5Fclose(fAx);
        H5Dclose(dAy);
        H5Fclose(fAy);
        H5Dclose(dAz);
        H5Fclose(fAz);
    }

    return 0;
}

static int write_rank_files_3d(const InputGridSpec *g,
                               const LaserPulse *pulse,
                               const char *prefix,
                               int compute_A,
                               size_t a2_i0, size_t a2_nloc,
                               int rank)
{
    // local dims: [t, ax1_global, ax2_local]
    hsize_t dims[3] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n, (hsize_t)a2_nloc};

    hid_t fEx, dEx, fEy, dEy, fEz, dEz;
    char path[512];

    component_filename(path, sizeof(path), prefix, "Ex", rank, 1);
    if (create_single_dataset_file(&fEx, &dEx, path, "Ex", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc) != 0)
        return 1;
    component_filename(path, sizeof(path), prefix, "Ey", rank, 1);
    if (create_single_dataset_file(&fEy, &dEy, path, "Ey", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc) != 0)
        return 2;
    component_filename(path, sizeof(path), prefix, "Ez", rank, 1);
    if (create_single_dataset_file(&fEz, &dEz, path, "Ez", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc) != 0)
        return 3;

    hid_t fAx = -1, dAx = -1, fAy = -1, dAy = -1, fAz = -1, dAz = -1;
    if (compute_A)
    {
        component_filename(path, sizeof(path), prefix, "Ax", rank, 1);
        if (create_single_dataset_file(&fAx, &dAx, path, "Ax", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc) != 0)
            return 4;
        component_filename(path, sizeof(path), prefix, "Ay", rank, 1);
        if (create_single_dataset_file(&fAy, &dAy, path, "Ay", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc) != 0)
            return 5;
        component_filename(path, sizeof(path), prefix, "Az", rank, 1);
        if (create_single_dataset_file(&fAz, &dAz, path, "Az", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc) != 0)
            return 6;
    }

    // one time-slice plane buffer per component
    size_t plane = (size_t)g->ax1_n * (size_t)a2_nloc;

    double *Ex = (double *)calloc(plane, sizeof(double));
    double *Ey = (double *)calloc(plane, sizeof(double));
    double *Ez = (double *)calloc(plane, sizeof(double));
    double *Ax = compute_A ? (double *)calloc(plane, sizeof(double)) : NULL;
    double *Ay = compute_A ? (double *)calloc(plane, sizeof(double)) : NULL;
    double *Az = compute_A ? (double *)calloc(plane, sizeof(double)) : NULL;

    if (!Ex || !Ey || !Ez || (compute_A && (!Ax || !Ay || !Az)))
        return 7;

    for (int it = 0; it < g->t_n; ++it)
    {
        double t = g->t_min + (double)it * g->dt;

        for (int i1 = 0; i1 < g->ax1_n; ++i1)
        {
            double a1 = g->ax1_min + (double)i1 * g->dx1;

            for (size_t jloc = 0; jloc < a2_nloc; ++jloc)
            {
                size_t jg = a2_i0 + jloc;
                double a2 = g->ax2_min + (double)jg * g->dx2;

                double r[3];
                set_r_from_axes(g, a1, a2, r);

                double E[3];
                LaserPulse_E(pulse, t, r, E);

                size_t idx = (size_t)i1 * a2_nloc + jloc;
                Ex[idx] = E[0];
                Ey[idx] = E[1];
                Ez[idx] = E[2];

                if (compute_A)
                {
                    double A[3];
                    LaserPulse_A(pulse, t, r, A);
                    Ax[idx] = A[0];
                    Ay[idx] = A[1];
                    Az[idx] = A[2];
                }
            }
        }

        // hyperslab [it, :, :]
        hsize_t start[3] = {(hsize_t)it, 0, 0};
        hsize_t count[3] = {1, (hsize_t)g->ax1_n, (hsize_t)a2_nloc};
        hid_t mspace = H5Screate_simple(3, count, NULL);

        hid_t fspace = H5Dget_space(dEx);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Dwrite(dEx, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, Ex);
        H5Sclose(fspace);

        fspace = H5Dget_space(dEy);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Dwrite(dEy, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, Ey);
        H5Sclose(fspace);

        fspace = H5Dget_space(dEz);
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
        H5Dwrite(dEz, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, Ez);
        H5Sclose(fspace);

        if (compute_A)
        {
            fspace = H5Dget_space(dAx);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
            H5Dwrite(dAx, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, Ax);
            H5Sclose(fspace);

            fspace = H5Dget_space(dAy);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
            H5Dwrite(dAy, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, Ay);
            H5Sclose(fspace);

            fspace = H5Dget_space(dAz);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
            H5Dwrite(dAz, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, Az);
            H5Sclose(fspace);
        }

        H5Sclose(mspace);
    }

    free(Ex);
    free(Ey);
    free(Ez);
    free(Ax);
    free(Ay);
    free(Az);

    H5Dclose(dEx);
    H5Fclose(fEx);
    H5Dclose(dEy);
    H5Fclose(fEy);
    H5Dclose(dEz);
    H5Fclose(fEz);

    if (compute_A)
    {
        H5Dclose(dAx);
        H5Fclose(fAx);
        H5Dclose(dAy);
        H5Fclose(fAy);
        H5Dclose(dAz);
        H5Fclose(fAz);
    }

    return 0;
}

static int merge_component_2d(const InputGridSpec *g, const char *prefix,
                              const char *comp, int compute_A,
                              int root, MPI_Comm comm)
{
    (void)compute_A;
    int rank = 0, nr = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nr);
    if (rank != root)
        return 0;

    // create global file for this component
    char outpath[512];
    component_filename(outpath, sizeof(outpath), prefix, comp, 0, 0);

    hsize_t gdims[2] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n};

    hid_t fout, dout;
    if (create_single_dataset_file(&fout, &dout, outpath, comp, 2, gdims, g, 0, 0, 0) != 0)
        return 1;

    for (int r = 0; r < nr; ++r)
    {
        char inpath[512];
        component_filename(inpath, sizeof(inpath), prefix, comp, r, 1);

        hid_t fin = h5_open_ro_or_fail(inpath);
        if (fin < 0)
        {
            fprintf(stderr, "merge: cannot open %s\n", inpath);
            continue;
        }

        // slab info
        int i0 = 0, nloc = 0;
        hid_t a = H5Aopen(fin, "slab_i0", H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_INT, &i0);
        H5Aclose(a);
        a = H5Aopen(fin, "slab_nloc", H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_INT, &nloc);
        H5Aclose(a);

        if (nloc <= 0)
        {
            H5Fclose(fin);
            continue;
        }

        hid_t din = H5Dopen2(fin, "/data", H5P_DEFAULT);
        if (din < 0)
        {
            H5Fclose(fin);
            continue;
        }

        double *buf = (double *)calloc((size_t)nloc, sizeof(double));
        if (!buf)
        {
            H5Dclose(din);
            H5Fclose(fin);
            continue;
        }

        for (int it = 0; it < g->t_n; ++it)
        {
            hsize_t s_start[2] = {(hsize_t)it, 0};
            hsize_t s_count[2] = {1, (hsize_t)nloc};
            hid_t smem = H5Screate_simple(2, s_count, NULL);

            // read from rank file
            hid_t ss = H5Dget_space(din);
            H5Sselect_hyperslab(ss, H5S_SELECT_SET, s_start, NULL, s_count, NULL);
            H5Dread(din, H5T_NATIVE_DOUBLE, smem, ss, H5P_DEFAULT, buf);
            H5Sclose(ss);

            // write into global
            hsize_t t_start[2] = {(hsize_t)it, (hsize_t)i0};
            hsize_t t_count[2] = {1, (hsize_t)nloc};
            hid_t ts = H5Dget_space(dout);
            H5Sselect_hyperslab(ts, H5S_SELECT_SET, t_start, NULL, t_count, NULL);
            H5Dwrite(dout, H5T_NATIVE_DOUBLE, smem, ts, H5P_DEFAULT, buf);
            H5Sclose(ts);

            H5Sclose(smem);
        }

        free(buf);
        H5Dclose(din);
        H5Fclose(fin);
    }

    H5Dclose(dout);
    H5Fclose(fout);
    return 0;
}

static int merge_component_3d(const InputGridSpec *g, const char *prefix,
                              const char *comp, int compute_A,
                              int root, MPI_Comm comm)
{
    (void)compute_A;
    int rank = 0, nr = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nr);
    if (rank != root)
        return 0;

    char outpath[512];
    component_filename(outpath, sizeof(outpath), prefix, comp, 0, 0);

    hsize_t gdims[3] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n, (hsize_t)g->ax2_n};

    hid_t fout, dout;
    if (create_single_dataset_file(&fout, &dout, outpath, comp, 3, gdims, g, 0, 0, 0) != 0)
        return 1;

    for (int r = 0; r < nr; ++r)
    {
        char inpath[512];
        component_filename(inpath, sizeof(inpath), prefix, comp, r, 1);

        hid_t fin = h5_open_ro_or_fail(inpath);
        if (fin < 0)
        {
            fprintf(stderr, "merge: cannot open %s\n", inpath);
            continue;
        }

        int i0 = 0, nloc = 0;
        hid_t a = H5Aopen(fin, "slab_i0", H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_INT, &i0);
        H5Aclose(a);
        a = H5Aopen(fin, "slab_nloc", H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_INT, &nloc);
        H5Aclose(a);

        if (nloc <= 0)
        {
            H5Fclose(fin);
            continue;
        }

        hid_t din = H5Dopen2(fin, "/data", H5P_DEFAULT);
        if (din < 0)
        {
            H5Fclose(fin);
            continue;
        }

        size_t plane = (size_t)g->ax1_n * (size_t)nloc;
        double *buf = (double *)calloc(plane, sizeof(double));
        if (!buf)
        {
            H5Dclose(din);
            H5Fclose(fin);
            continue;
        }

        for (int it = 0; it < g->t_n; ++it)
        {
            hsize_t s_start[3] = {(hsize_t)it, 0, 0};
            hsize_t s_count[3] = {1, (hsize_t)g->ax1_n, (hsize_t)nloc};
            hid_t smem = H5Screate_simple(3, s_count, NULL);

            hid_t ss = H5Dget_space(din);
            H5Sselect_hyperslab(ss, H5S_SELECT_SET, s_start, NULL, s_count, NULL);
            H5Dread(din, H5T_NATIVE_DOUBLE, smem, ss, H5P_DEFAULT, buf);
            H5Sclose(ss);

            hsize_t t_start[3] = {(hsize_t)it, 0, (hsize_t)i0};
            hsize_t t_count[3] = {1, (hsize_t)g->ax1_n, (hsize_t)nloc};
            hid_t ts = H5Dget_space(dout);
            H5Sselect_hyperslab(ts, H5S_SELECT_SET, t_start, NULL, t_count, NULL);
            H5Dwrite(dout, H5T_NATIVE_DOUBLE, smem, ts, H5P_DEFAULT, buf);
            H5Sclose(ts);

            H5Sclose(smem);
        }

        free(buf);
        H5Dclose(din);
        H5Fclose(fin);
    }

    H5Dclose(dout);
    H5Fclose(fout);
    return 0;
}

int field_cache_run(const InputSimSpec *sim,
                    const LaserPulse *pulse,
                    const char *prefix,
                    const FieldCacheOptions *opt_in,
                    MPI_Comm comm)
{
    if (!sim || !pulse || !prefix)
        return 1;

    FieldCacheOptions opt = opt_in ? *opt_in : field_cache_default_options();

    int rank = 0, nr = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nr);

    if (rank == opt.root_rank)
    {
        int mk = ensure_parent_dir(prefix);
        if (mk != 0)
        {
            fprintf(stderr, "field_cache: failed to create output directory for prefix \"%s\" (rc=%d)\n", prefix, mk);
            // abort cleanly for all ranks
            MPI_Abort(comm, 100);
        }
    }

    MPI_Barrier(comm);

    (void)nr;

    const InputGridSpec *g = &sim->grid;

    status_rootf(comm, opt.root_rank,
                 "field_cache: time grid t_min=%f, t_max=%f, dt=%f t_n=%d",
                 g->t_min, g->t_max, g->dt, g->t_n);
    status_rootf(comm, opt.root_rank,
                 "field_cache: spatial grid x1_min=%f, x1_max=%f, dx1=%f, x1_n=%d",
                 g->ax1_min, g->ax1_max, g->dx1, g->ax1_n);
    if (g->has_ax2)
    {
        status_rootf(comm, opt.root_rank,
                     "field_cache: spatial grid x2_min=%f, x2_max=%f, dx2=%f, x2_n=%d",
                     g->ax2_min, g->ax2_max, g->dx2, g->ax2_n);
    }
    status_rootf(comm, opt.root_rank, "field_cache: ranks=%d, output prefix=\"%s\"", nr, prefix);

    int rc = 0;

    if (!g->has_ax2)
    {
        // 2D total: decompose along ax1
        size_t i0 = 0, nloc = 0;
        decompose_1d((size_t)g->ax1_n, rank, nr, &i0, &nloc);

        rc = write_rank_files_2d(g, pulse, prefix, opt.compute_A, i0, nloc, rank);
        if (rc != 0)
        {
            die_root(comm, opt.root_rank, "field_cache: rank write (2D) failed");
            return 10 + rc;
        }

        MPI_Barrier(comm);

        if (opt.merge_on_root)
        {
            status_root(comm, opt.root_rank, "field_cache: per-rank slabs computed");

            rc = merge_component_2d(g, prefix, "Ex", opt.compute_A, opt.root_rank, comm);
            if (rc == 0)
                remove_rank_files(prefix, "Ex", nr, opt.root_rank, comm);
            status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ex");
            if (rc != 0)
                return 20 + rc;

            rc = merge_component_2d(g, prefix, "Ey", opt.compute_A, opt.root_rank, comm);
            if (rc == 0)
                remove_rank_files(prefix, "Ey", nr, opt.root_rank, comm);
            status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ey");
            if (rc != 0)
                return 21 + rc;

            rc = merge_component_2d(g, prefix, "Ez", opt.compute_A, opt.root_rank, comm);
            if (rc == 0)
                remove_rank_files(prefix, "Ez", nr, opt.root_rank, comm);
            status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ez");
            if (rc != 0)
                return 22 + rc;

            if (opt.compute_A)
            {
                rc = merge_component_2d(g, prefix, "Ax", opt.compute_A, opt.root_rank, comm);
                if (rc == 0)
                    remove_rank_files(prefix, "Ax", nr, opt.root_rank, comm);
                status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ax");
                if (rc != 0)
                    return 23 + rc;

                rc = merge_component_2d(g, prefix, "Ay", opt.compute_A, opt.root_rank, comm);
                if (rc == 0)
                    remove_rank_files(prefix, "Ay", nr, opt.root_rank, comm);
                status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ay");
                if (rc != 0)
                    return 24 + rc;

                rc = merge_component_2d(g, prefix, "Az", opt.compute_A, opt.root_rank, comm);
                if (rc == 0)
                    remove_rank_files(prefix, "Az", nr, opt.root_rank, comm);
                status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Az");
                if (rc != 0)
                    return 25 + rc;
            }
            MPI_Barrier(comm);
        }
        return 0;
    }
    status_root(comm, opt.root_rank, "field_cache: computing per-rank slabs...");

    // 3D total: decompose along ax2
    size_t i0 = 0, nloc = 0;
    decompose_1d((size_t)g->ax2_n, rank, nr, &i0, &nloc);

    rc = write_rank_files_3d(g, pulse, prefix, opt.compute_A, i0, nloc, rank);
    if (rc != 0)
    {
        die_root(comm, opt.root_rank, "field_cache: rank write (3D) failed");
        return 30 + rc;
    }

    MPI_Barrier(comm);
    status_root(comm, opt.root_rank, "field_cache: per-rank slabs computed");

    if (opt.merge_on_root)
    {
        status_root(comm, opt.root_rank, "field_cache: merging rank slabs into final files...");
        rc = merge_component_3d(g, prefix, "Ex", opt.compute_A, opt.root_rank, comm);
        remove_rank_files(prefix, "Ex", nr, opt.root_rank, comm);
        if (rc != 0)
            return 40 + rc;
        status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ex");

        rc = merge_component_3d(g, prefix, "Ey", opt.compute_A, opt.root_rank, comm);
        remove_rank_files(prefix, "Ey", nr, opt.root_rank, comm);
        if (rc != 0)
            return 41 + rc;
        status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ey");

        rc = merge_component_3d(g, prefix, "Ez", opt.compute_A, opt.root_rank, comm);
        remove_rank_files(prefix, "Ez", nr, opt.root_rank, comm);
        if (rc != 0)
            return 42 + rc;
        status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ez");

        if (opt.compute_A)
        {
            rc = merge_component_3d(g, prefix, "Ax", opt.compute_A, opt.root_rank, comm);
            remove_rank_files(prefix, "Ax", nr, opt.root_rank, comm);
            if (rc != 0)
                return 43 + rc;
            status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ax");

            rc = merge_component_3d(g, prefix, "Ay", opt.compute_A, opt.root_rank, comm);
            remove_rank_files(prefix, "Ay", nr, opt.root_rank, comm);
            if (rc != 0)
                return 44 + rc;
            status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Ay");

            rc = merge_component_3d(g, prefix, "Az", opt.compute_A, opt.root_rank, comm);
            remove_rank_files(prefix, "Az", nr, opt.root_rank, comm);
            if (rc != 0)
                return 45 + rc;
            status_rootf(comm, opt.root_rank, "field_cache: merged %s", "Az");
        }
    }
    status_root(comm, opt.root_rank, "field_cache: done");
    return 0;
}
