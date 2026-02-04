/* field_cache.c */

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
#include <inttypes.h>

/* -------------------------- small status helpers -------------------------- */

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
    char tmp[512];
    size_t n = strlen(prefix);
    if (n >= sizeof(tmp))
        return 1;

    strcpy(tmp, prefix);

    char *slash = strrchr(tmp, '/');
    if (!slash)
        return 0;

    *slash = '\0';

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

/* -------------------------- options -------------------------- */

FieldCacheOptions field_cache_default_options(void)
{
    FieldCacheOptions o;
    o.compute_A = 1;
    o.merge_on_root = 1;
    o.root_rank = 0;
    return o;
}

/* -------------------------- axis helpers -------------------------- */

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
    case AXIS_X: r_um[0] = a1; break;
    case AXIS_Y: r_um[1] = a1; break;
    case AXIS_Z: r_um[2] = a1; break;
    case AXIS_INVALID:
    default:
        fprintf(stderr, "field_cache: invalid axis in grid spec\n");
        break;
    }

    if (g->has_ax2)
    {
        switch (g->ax2)
        {
        case AXIS_X: r_um[0] = a2; break;
        case AXIS_Y: r_um[1] = a2; break;
        case AXIS_Z: r_um[2] = a2; break;
        case AXIS_INVALID:
        default:
            fprintf(stderr, "field_cache: invalid axis in grid spec\n");
            break;
        }
    }
}

static void decompose_1d(size_t n, int rank, int nranks, size_t *i0, size_t *nloc)
{
    size_t base = n / (size_t)nranks;
    size_t rem  = n % (size_t)nranks;

    size_t start = (size_t)rank * base + ((rank < (int)rem) ? (size_t)rank : rem);

    size_t count = base + (size_t)((rank < (int)rem) ? 1 : 0);

    *i0   = start;
    *nloc = count;
}

/* -------------------------- HDF5 open/create -------------------------- */

static hid_t h5_create_or_fail(const char *path)
{
    return H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

static hid_t h5_open_ro_or_fail(const char *path)
{
    return H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
}

/* -------------------------- simple attribute writers -------------------------- */

static int h5_write_attr_string(hid_t obj, const char *name, const char *value)
{
    if (!value) value = "";

    hid_t atype = H5Tcopy(H5T_C_S1);
    if (atype < 0)
        return 1;

    if (H5Tset_size(atype, strlen(value)) < 0)
    {
        H5Tclose(atype);
        return 2;
    }

    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
    {
        H5Tclose(atype);
        return 3;
    }

    hid_t attr = H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        H5Tclose(atype);
        return 4;
    }

    herr_t st = H5Awrite(attr, atype, value);
    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);

    return (st < 0) ? 5 : 0;
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

/* --- OSIRIS-style attribute helpers (1-element arrays and string arrays) --- */
static int h5_write_attr_str_array(hid_t obj, const char *name, int n, const char *const *vals)
{
    hid_t dtype = H5Tcopy(H5T_C_S1);
    if (dtype < 0)
        return 1;
    if (H5Tset_size(dtype, H5T_VARIABLE) < 0)
    {
        H5Tclose(dtype);
        return 2;
    }

    hsize_t dims[1] = {(hsize_t)n};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
    {
        H5Tclose(dtype);
        return 3;
    }

    hid_t attr = H5Acreate2(obj, name, dtype, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        H5Tclose(dtype);
        return 4;
    }

    herr_t st = H5Awrite(attr, dtype, vals);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(dtype);

    return (st < 0) ? 5 : 0;
}

static int h5_write_attr_string1(hid_t obj, const char *name, const char *val)
{
    const char *a[1] = {val};
    return h5_write_attr_str_array(obj, name, 1, a);
}

static int h5_write_attr_double_array(hid_t obj, const char *name, int n, const double *v)
{
    hsize_t dims[1] = {(hsize_t)n};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
        return 1;

    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        return 2;
    }

    herr_t st = H5Awrite(attr, H5T_NATIVE_DOUBLE, v);
    H5Aclose(attr);
    H5Sclose(space);

    return (st < 0) ? 3 : 0;
}

static int h5_write_attr_double1(hid_t obj, const char *name, double v)
{
    double a[1] = {v};
    return h5_write_attr_double_array(obj, name, 1, a);
}

static int h5_write_attr_int_array(hid_t obj, const char *name, int n, const int *v)
{
    hsize_t dims[1] = {(hsize_t)n};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
        return 1;

    hid_t attr = H5Acreate2(obj, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(space);
        return 2;
    }

    herr_t st = H5Awrite(attr, H5T_NATIVE_INT, v);
    H5Aclose(attr);
    H5Sclose(space);

    return (st < 0) ? 3 : 0;
}

static int h5_write_attr_int1(hid_t obj, const char *name, int v)
{
    int a[1] = {v};
    return h5_write_attr_int_array(obj, name, 1, a);
}

/* ----------------------- cache signature (grid+laser) ----------------------- */

typedef struct
{
    char *buf;
    size_t len;
    size_t cap;
} StrBuf;

static void sb_init(StrBuf *s) { s->buf = NULL; s->len = 0; s->cap = 0; }
static void sb_free(StrBuf *s) { free(s->buf); s->buf = NULL; s->len = 0; s->cap = 0; }

static int sb_ensure(StrBuf *s, size_t need_extra)
{
    size_t need = s->len + need_extra + 1;
    if (need <= s->cap) return 0;

    size_t newcap = (s->cap == 0) ? 1024 : s->cap;
    while (newcap < need) newcap *= 2;

    char *p = (char *)realloc(s->buf, newcap);
    if (!p) return 1;

    s->buf = p;
    s->cap = newcap;
    return 0;
}

static int sb_append(StrBuf *s, const char *txt)
{
    if (!txt) txt = "";
    size_t n = strlen(txt);
    if (sb_ensure(s, n) != 0) return 1;
    memcpy(s->buf + s->len, txt, n);
    s->len += n;
    s->buf[s->len] = '\0';
    return 0;
}

static int sb_appendf(StrBuf *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);

    if (n < 0)
    {
        va_end(ap);
        return 1;
    }

    if (sb_ensure(s, (size_t)n) != 0)
    {
        va_end(ap);
        return 2;
    }

    vsnprintf(s->buf + s->len, s->cap - s->len, fmt, ap);
    va_end(ap);

    s->len += (size_t)n;
    return 0;
}

static uint64_t fnv1a64(const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;

    for (size_t i = 0; i < n; ++i)
    {
        h ^= (uint64_t)p[i];
        h *= prime;
    }
    return h;
}

static char *field_cache_build_config_string(const InputSimSpec *sim, int compute_A)
{
    if (!sim) return NULL;

    const InputGridSpec *g = &sim->grid;

    StrBuf s;
    sb_init(&s);

    if (sb_append(&s, "FIELD_CACHE_CONFIG v1\n") != 0) goto oom;
    if (sb_appendf(&s, "compute_A=%d\n", compute_A ? 1 : 0) != 0) goto oom;

    if (sb_append(&s, "[grid]\n") != 0) goto oom;
    if (sb_appendf(&s, "t_min=%.17g\n", g->t_min) != 0) goto oom;
    if (sb_appendf(&s, "t_max=%.17g\n", g->t_max) != 0) goto oom;
    if (sb_appendf(&s, "dt=%.17g\n", g->dt) != 0) goto oom;
    if (sb_appendf(&s, "t_n=%d\n", g->t_n) != 0) goto oom;

    if (sb_appendf(&s, "ax1=%s\n", axis_name(g->ax1)) != 0) goto oom;
    if (sb_appendf(&s, "has_ax2=%d\n", g->has_ax2 ? 1 : 0) != 0) goto oom;
    if (g->has_ax2)
        if (sb_appendf(&s, "ax2=%s\n", axis_name(g->ax2)) != 0) goto oom;

    if (sb_appendf(&s, "fixed_x=%.17g\n", g->fixed_x) != 0) goto oom;
    if (sb_appendf(&s, "fixed_y=%.17g\n", g->fixed_y) != 0) goto oom;
    if (sb_appendf(&s, "fixed_z=%.17g\n", g->fixed_z) != 0) goto oom;

    if (sb_appendf(&s, "ax1_min=%.17g\n", g->ax1_min) != 0) goto oom;
    if (sb_appendf(&s, "ax1_max=%.17g\n", g->ax1_max) != 0) goto oom;
    if (sb_appendf(&s, "dx1=%.17g\n", g->dx1) != 0) goto oom;
    if (sb_appendf(&s, "ax1_n=%d\n", g->ax1_n) != 0) goto oom;

    if (g->has_ax2)
    {
        if (sb_appendf(&s, "ax2_min=%.17g\n", g->ax2_min) != 0) goto oom;
        if (sb_appendf(&s, "ax2_max=%.17g\n", g->ax2_max) != 0) goto oom;
        if (sb_appendf(&s, "dx2=%.17g\n", g->dx2) != 0) goto oom;
        if (sb_appendf(&s, "ax2_n=%d\n", g->ax2_n) != 0) goto oom;
    }

    if (sb_append(&s, "[lasers]\n") != 0) goto oom;
    if (sb_appendf(&s, "count=%zu\n", sim->lasers.count) != 0) goto oom;

    for (size_t i = 0; i < sim->lasers.count; ++i)
    {
        const InputLaserSpec *L = &sim->lasers.items[i];

        if (sb_appendf(&s, "laser[%zu].type=%d\n", i, (int)L->type) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].E0=%.17g\n", i, L->E0) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].wavelength=%.17g\n", i, L->wavelength) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].phase0=%.17g\n", i, L->phase0) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].k_vec=%.17g,%.17g,%.17g\n",
                       i, L->k_vec[0], L->k_vec[1], L->k_vec[2]) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].r_start=%.17g,%.17g,%.17g\n",
                       i, L->r_start[0], L->r_start[1], L->r_start[2]) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].use_retarded_time=%d\n",
                       i, L->use_retarded_time ? 1 : 0) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].temporal_type=%d\n", i, (int)L->temporal_type) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].tau=%.17g\n", i, L->tau) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].transverse_type=%d\n", i, (int)L->transverse_type) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].w0=%.17g\n", i, L->w0) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].zf=%.17g\n", i, L->zf) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].has_hermite=%d\n", i, L->has_hermite ? 1 : 0) != 0) goto oom;
        if (L->has_hermite)
            if (sb_appendf(&s, "laser[%zu].herm_lm=%d,%d\n", i, L->herm_l, L->herm_m) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].polarization=%d\n", i, (int)L->polarization) != 0) goto oom;
        if (sb_appendf(&s, "laser[%zu].angle=%.17g\n", i, L->angle) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].sense=%d\n", i, (int)L->sense) != 0) goto oom;

        if (sb_appendf(&s, "laser[%zu].has_jones=%d\n", i, L->has_jones ? 1 : 0) != 0) goto oom;
        if (L->has_jones)
        {
            if (sb_appendf(&s, "laser[%zu].p1=%.17g\n", i, L->p1) != 0) goto oom;
            if (sb_appendf(&s, "laser[%zu].p2=%.17g\n", i, L->p2) != 0) goto oom;
            if (sb_appendf(&s, "laser[%zu].delta=%.17g\n", i, L->delta) != 0) goto oom;
        }
    }

    return s.buf;

oom:
    sb_free(&s);
    return NULL;
}

/* -------------------------- component filename -------------------------- */

static void component_filename(char *buf, size_t bufsz,
                               const char *cache_dir,
                               const char *comp, int rank, int is_rank)
{
    char fname[128];
    if (is_rank)
        snprintf(fname, sizeof(fname), "%s.rank%04d.h5", comp, rank);
    else
        snprintf(fname, sizeof(fname), "%s.h5", comp);

    size_t dir_len = strlen(cache_dir);
    int need_slash = (dir_len > 0 && cache_dir[dir_len - 1] != '/');

    int n = snprintf(buf, bufsz, "%s%s%s", cache_dir, need_slash ? "/" : "", fname);
    if (n < 0 || (size_t)n >= bufsz)
    {
        fprintf(stderr, "field_cache: component path too long for dir='%s' comp='%s'\n", cache_dir, comp);
        abort();
    }
}

/* -------------------------- OSIRIS axis datasets -------------------------- */

static const char *axis_name_osiris(Axis a)
{
    switch (a)
    {
    case AXIS_X: return "x";
    case AXIS_Y: return "y";
    case AXIS_Z: return "z";
    default:     return "?";
    }
}

static const char *axis_long_name_osiris(Axis a)
{
    switch (a)
    {
    case AXIS_X: return "x";
    case AXIS_Y: return "y";
    case AXIS_Z: return "z";
    default:     return "?";
    }
}

static int osiris_write_axis_dataset(hid_t axis_group, int idx1, const char *name, const char *long_name,
                                     const char *units, double vmin, double vmax)
{
    char dname[32];
    snprintf(dname, sizeof(dname), "AXIS%d", idx1);

    double vv[2] = {vmin, vmax};
    hsize_t dims[1] = {2};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
        return 1;

    hid_t dset = H5Dcreate2(axis_group, dname, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);
    if (dset < 0)
        return 2;

    herr_t st = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vv);
    if (st < 0)
    {
        H5Dclose(dset);
        return 3;
    }

    h5_write_attr_string1(dset, "NAME", name);
    h5_write_attr_string1(dset, "LONG_NAME", long_name);
    h5_write_attr_string1(dset, "TYPE", "linear");
    h5_write_attr_string1(dset, "UNITS", units);

    H5Dclose(dset);
    return 0;
}

/* -------------------------- file creation (single dataset) -------------------------- */

static int create_single_dataset_file(hid_t *out_f, hid_t *out_dset,
                                      const char *path,
                                      const char *component_name,
                                      int nd, const hsize_t *dims,
                                      const InputGridSpec *g,
                                      int is_rank_file,
                                      int ax_i0, int ax_nloc,
                                      const char *cache_config,
                                      const char *cache_key,
                                      int cache_has_A)
{
    hid_t f = h5_create_or_fail(path);
    if (f < 0)
        return 1;

    /* --- OSIRIS-compatible structure (AXIS + SIMULATION + root attrs) --- */
    {
        h5_write_attr_string1(f, "TYPE", "grid");
        h5_write_attr_string1(f, "NAME", component_name);
        h5_write_attr_string1(f, "LABEL", component_name);

        if (component_name && component_name[0] == 'E')
            h5_write_attr_string1(f, "UNITS", "GV/m");
        else if (component_name && component_name[0] == 'A')
            h5_write_attr_string1(f, "UNITS", "GV/m fs");
        else
            h5_write_attr_string1(f, "UNITS", "a.u.");

        h5_write_attr_int1(f, "ITER", 0);
        h5_write_attr_double1(f, "TIME", 0.0);
        h5_write_attr_string1(f, "TIME UNITS", "fs");
        h5_write_attr_double1(f, "OFFSET_T", 0.0);

        {
            double offx[3] = {0.0, 0.0, 0.0};
            h5_write_attr_double_array(f, "OFFSET_X", nd, offx);
        }

        hid_t g_axis = H5Gcreate2(f, "AXIS", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (g_axis >= 0)
        {
            if (nd == 2)
            {
                osiris_write_axis_dataset(g_axis, 1,
                                          axis_name_osiris(g->ax1), axis_long_name_osiris(g->ax1),
                                          "\\mu m", g->ax1_min, g->ax1_max);

                osiris_write_axis_dataset(g_axis, 2, "t", "t", "fs", g->t_min, g->t_max);
            }
            else
            {
                osiris_write_axis_dataset(g_axis, 1,
                                          axis_name_osiris(g->ax2), axis_long_name_osiris(g->ax2),
                                          "\\mu m", g->ax2_min, g->ax2_max);

                osiris_write_axis_dataset(g_axis, 2,
                                          axis_name_osiris(g->ax1), axis_long_name_osiris(g->ax1),
                                          "\\mu m", g->ax1_min, g->ax1_max);

                osiris_write_axis_dataset(g_axis, 3, "t", "t", "fs", g->t_min, g->t_max);
            }

            H5Gclose(g_axis);
        }

        hid_t g_sim = H5Gcreate2(f, "SIMULATION", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (g_sim >= 0)
        {
            int ndims_arr[1] = {nd};
            h5_write_attr_int_array(g_sim, "NDIMS", 1, ndims_arr);

            {
                int nx[3] = {0, 0, 0};

                if (nd == 2)
                {
                    nx[0] = (int)dims[1];
                    nx[1] = (int)dims[0];
                }
                else
                {
                    nx[0] = (int)dims[2];
                    nx[1] = (int)dims[1];
                    nx[2] = (int)dims[0];
                }

                h5_write_attr_int_array(g_sim, "NX", nd, nx);
            }

            {
                int nx[3] = {1, 1, 1};
                h5_write_attr_int_array(g_sim, "PAR_NODE_CONF", nd, nx);
            }

            h5_write_attr_double_array(g_sim, "DT", 1, (double[]){g->dt});

            {
                double xmin[3] = {0.0, 0.0, 0.0};
                double xmax[3] = {0.0, 0.0, 0.0};

                if (nd == 2)
                {
                    xmin[0] = g->ax1_min;
                    xmax[0] = g->ax1_max;
                    xmin[1] = g->t_min;
                    xmax[1] = g->t_max;
                }
                else
                {
                    xmin[0] = g->ax2_min;
                    xmax[0] = g->ax2_max;
                    xmin[1] = g->ax1_min;
                    xmax[1] = g->ax1_max;
                    xmin[2] = g->t_min;
                    xmax[2] = g->t_max;
                }

                h5_write_attr_double_array(g_sim, "XMIN", nd, xmin);
                h5_write_attr_double_array(g_sim, "XMAX", nd, xmax);
            }

            H5Gclose(g_sim);
        }
    }

    /* metadata */
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

    /* cache provenance (NEW) */
    if (cache_config) h5_write_attr_string(f, "CACHE_CONFIG", cache_config);
    if (cache_key)    h5_write_attr_string(f, "CACHE_KEY", cache_key);
    h5_write_attr_int(f, "CACHE_HAS_A", cache_has_A ? 1 : 0);

    /* single dataset */
    hsize_t cdims[3];
    const hsize_t *use_dims = dims;

    if (nd == 3)
    {
        if (dims[0] != (hsize_t)g->t_n && dims[2] == (hsize_t)g->t_n)
        {
            cdims[0] = dims[2];
            cdims[1] = dims[0];
            cdims[2] = dims[1];
            use_dims  = cdims;
        }
    }
    else if (nd == 2)
    {
        if (dims[0] != (hsize_t)g->t_n && dims[1] == (hsize_t)g->t_n)
        {
            cdims[0] = dims[1];
            cdims[1] = dims[0];
            use_dims  = cdims;
        }
    }

    hid_t space = H5Screate_simple(nd, use_dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 2;
    }

    char dset_path[64];
    snprintf(dset_path, sizeof(dset_path), "/%s", component_name);

    hid_t dset = H5Dcreate2(f, dset_path, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);

    if (dset < 0)
    {
        H5Fclose(f);
        return 3;
    }

    *out_f    = f;
    *out_dset = dset;
    return 0;
}

/* -------------------------- cleanup rank files -------------------------- */

static void remove_rank_files(const char *cache_dir, const char *comp, int nranks, int root, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != root)
        return;

    char path[512];
    for (int r = 0; r < nranks; ++r)
    {
        component_filename(path, sizeof(path), cache_dir, comp, r, 1);
        if (unlink(path) != 0)
            fprintf(stderr, "field_cache: warning: could not remove partial file \"%s\"\n", path);
    }
}

/* -------------------------- rank writers -------------------------- */

static int write_rank_files_2d(const InputGridSpec *g,
                               const LaserPulse *pulse,
                               const char *prefix,
                               int compute_A,
                               size_t a1_i0, size_t a1_nloc,
                               int rank,
                               const char *cache_config,
                               const char *cache_key,
                               int cache_has_A)
{
    hsize_t dims[2] = {(hsize_t)g->t_n, (hsize_t)a1_nloc};

    hid_t fEx, dEx, fEy, dEy, fEz, dEz;
    char path[512];

    component_filename(path, sizeof(path), prefix, "Ex", rank, 1);
    if (create_single_dataset_file(&fEx, &dEx, path, "Ex", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc,
                                   cache_config, cache_key, cache_has_A) != 0)
        return 1;

    component_filename(path, sizeof(path), prefix, "Ey", rank, 1);
    if (create_single_dataset_file(&fEy, &dEy, path, "Ey", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc,
                                   cache_config, cache_key, cache_has_A) != 0)
        return 2;

    component_filename(path, sizeof(path), prefix, "Ez", rank, 1);
    if (create_single_dataset_file(&fEz, &dEz, path, "Ez", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc,
                                   cache_config, cache_key, cache_has_A) != 0)
        return 3;

    hid_t fAx = -1, dAx = -1, fAy = -1, dAy = -1, fAz = -1, dAz = -1;
    if (compute_A)
    {
        component_filename(path, sizeof(path), prefix, "Ax", rank, 1);
        if (create_single_dataset_file(&fAx, &dAx, path, "Ax", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc,
                                       cache_config, cache_key, cache_has_A) != 0)
            return 4;

        component_filename(path, sizeof(path), prefix, "Ay", rank, 1);
        if (create_single_dataset_file(&fAy, &dAy, path, "Ay", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc,
                                       cache_config, cache_key, cache_has_A) != 0)
            return 5;

        component_filename(path, sizeof(path), prefix, "Az", rank, 1);
        if (create_single_dataset_file(&fAz, &dAz, path, "Az", 2, dims, g, 1, (int)a1_i0, (int)a1_nloc,
                                       cache_config, cache_key, cache_has_A) != 0)
            return 6;
    }

    double *bx  = (double *)calloc(a1_nloc, sizeof(double));
    double *by  = (double *)calloc(a1_nloc, sizeof(double));
    double *bz  = (double *)calloc(a1_nloc, sizeof(double));
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

    free(bx); free(by); free(bz);
    free(bax); free(bay); free(baz);

    H5Dclose(dEx); H5Fclose(fEx);
    H5Dclose(dEy); H5Fclose(fEy);
    H5Dclose(dEz); H5Fclose(fEz);

    if (compute_A)
    {
        H5Dclose(dAx); H5Fclose(fAx);
        H5Dclose(dAy); H5Fclose(fAy);
        H5Dclose(dAz); H5Fclose(fAz);
    }

    return 0;
}

static int write_rank_files_3d(const InputGridSpec *g,
                               const LaserPulse *pulse,
                               const char *prefix,
                               int compute_A,
                               size_t a2_i0, size_t a2_nloc,
                               int rank,
                               const char *cache_config,
                               const char *cache_key,
                               int cache_has_A)
{
    hsize_t dims[3] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n, (hsize_t)a2_nloc};

    hid_t fEx, dEx, fEy, dEy, fEz, dEz;
    char path[512];

    component_filename(path, sizeof(path), prefix, "Ex", rank, 1);
    if (create_single_dataset_file(&fEx, &dEx, path, "Ex", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc,
                                   cache_config, cache_key, cache_has_A) != 0)
        return 1;

    component_filename(path, sizeof(path), prefix, "Ey", rank, 1);
    if (create_single_dataset_file(&fEy, &dEy, path, "Ey", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc,
                                   cache_config, cache_key, cache_has_A) != 0)
        return 2;

    component_filename(path, sizeof(path), prefix, "Ez", rank, 1);
    if (create_single_dataset_file(&fEz, &dEz, path, "Ez", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc,
                                   cache_config, cache_key, cache_has_A) != 0)
        return 3;

    hid_t fAx = -1, dAx = -1, fAy = -1, dAy = -1, fAz = -1, dAz = -1;
    if (compute_A)
    {
        component_filename(path, sizeof(path), prefix, "Ax", rank, 1);
        if (create_single_dataset_file(&fAx, &dAx, path, "Ax", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc,
                                       cache_config, cache_key, cache_has_A) != 0)
            return 4;

        component_filename(path, sizeof(path), prefix, "Ay", rank, 1);
        if (create_single_dataset_file(&fAy, &dAy, path, "Ay", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc,
                                       cache_config, cache_key, cache_has_A) != 0)
            return 5;

        component_filename(path, sizeof(path), prefix, "Az", rank, 1);
        if (create_single_dataset_file(&fAz, &dAz, path, "Az", 3, dims, g, 1, (int)a2_i0, (int)a2_nloc,
                                       cache_config, cache_key, cache_has_A) != 0)
            return 6;
    }

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

    free(Ex); free(Ey); free(Ez);
    free(Ax); free(Ay); free(Az);

    H5Dclose(dEx); H5Fclose(fEx);
    H5Dclose(dEy); H5Fclose(fEy);
    H5Dclose(dEz); H5Fclose(fEz);

    if (compute_A)
    {
        H5Dclose(dAx); H5Fclose(fAx);
        H5Dclose(dAy); H5Fclose(fAy);
        H5Dclose(dAz); H5Fclose(fAz);
    }

    return 0;
}

/* -------------------------- merge helpers -------------------------- */

static int merge_component_2d(const InputGridSpec *g, const char *prefix,
                              const char *comp, int compute_A,
                              int root, MPI_Comm comm,
                              const char *cache_config,
                              const char *cache_key,
                              int cache_has_A)
{
    (void)compute_A;

    int rank = 0, nr = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nr);

    if (rank != root)
        return 0;

    char outpath[512];
    component_filename(outpath, sizeof(outpath), prefix, comp, 0, 0);

    hsize_t gdims[2] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n};

    hid_t fout, dout;
    if (create_single_dataset_file(&fout, &dout, outpath, comp, 2, gdims, g, 0, 0, 0,
                                   cache_config, cache_key, cache_has_A) != 0)
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
        if (a >= 0) { H5Aread(a, H5T_NATIVE_INT, &i0); H5Aclose(a); }
        a = H5Aopen(fin, "slab_nloc", H5P_DEFAULT);
        if (a >= 0) { H5Aread(a, H5T_NATIVE_INT, &nloc); H5Aclose(a); }

        if (nloc <= 0)
        {
            H5Fclose(fin);
            continue;
        }

        char dset_path[64];
        snprintf(dset_path, sizeof(dset_path), "/%s", comp);

        hid_t din = H5Dopen2(fin, dset_path, H5P_DEFAULT);
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

            hid_t ss = H5Dget_space(din);
            H5Sselect_hyperslab(ss, H5S_SELECT_SET, s_start, NULL, s_count, NULL);
            H5Dread(din, H5T_NATIVE_DOUBLE, smem, ss, H5P_DEFAULT, buf);
            H5Sclose(ss);

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
                              int root, MPI_Comm comm,
                              const char *cache_config,
                              const char *cache_key,
                              int cache_has_A)
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
    if (create_single_dataset_file(&fout, &dout, outpath, comp, 3, gdims, g, 0, 0, 0,
                                   cache_config, cache_key, cache_has_A) != 0)
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
        if (a >= 0) { H5Aread(a, H5T_NATIVE_INT, &i0); H5Aclose(a); }
        a = H5Aopen(fin, "slab_nloc", H5P_DEFAULT);
        if (a >= 0) { H5Aread(a, H5T_NATIVE_INT, &nloc); H5Aclose(a); }

        if (nloc <= 0)
        {
            H5Fclose(fin);
            continue;
        }

        char dset_path[64];
        snprintf(dset_path, sizeof(dset_path), "/%s", comp);

        hid_t din = H5Dopen2(fin, dset_path, H5P_DEFAULT);
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

/* -------------------------- compatibility check (NEW) -------------------------- */

static int h5_read_attr_int(hid_t obj, const char *name, int *out)
{
    if (!out) return -1;

    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) return 0;

    herr_t st = H5Aread(a, H5T_NATIVE_INT, out);
    H5Aclose(a);

    return (st < 0) ? -2 : 1;
}

/* Read fixed-length string attribute as allocated C string. Works with your h5_write_attr_string(). */
static int h5_read_attr_string_alloc(hid_t obj, const char *name, char **out_s)
{
    if (!out_s) return -1;
    *out_s = NULL;

    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) return 0;

    hid_t t = H5Aget_type(a);
    if (t < 0) { H5Aclose(a); return -2; }

    size_t sz = (size_t)H5Tget_size(t);
    if (sz == 0) { H5Tclose(t); H5Aclose(a); return -3; }

    char *buf = (char *)malloc(sz + 1);
    if (!buf) { H5Tclose(t); H5Aclose(a); return -4; }

    herr_t st = H5Aread(a, t, buf);
    H5Tclose(t);
    H5Aclose(a);

    if (st < 0)
    {
        free(buf);
        return -5;
    }

    buf[sz] = '\0';
    *out_s = buf;
    return 1;
}

/* Public: return 1 if compatible, 0 if not, <0 error. */
int field_cache_is_compatible(const char *cache_dir, const InputSimSpec *sim, int require_A,
                              char *why, size_t why_sz)
{
    if (why && why_sz) why[0] = '\0';
    if (!cache_dir || !sim) return -1;

    /* Open merged Ex.h5 (we use a single component as authority) */
    char path[512];
    component_filename(path, sizeof(path), cache_dir, "Ex", 0, 0);

    hid_t f = h5_open_ro_or_fail(path);
    if (f < 0)
    {
        if (why && why_sz) snprintf(why, why_sz, "cannot open \"%s\"", path);
        return 0;
    }

    char *key_file = NULL;
    int hasA_file = 0;

    int rkey = h5_read_attr_string_alloc(f, "CACHE_KEY", &key_file);
    int rA   = h5_read_attr_int(f, "CACHE_HAS_A", &hasA_file);

    H5Fclose(f);

    if (rkey <= 0 || !key_file)
    {
        if (why && why_sz) snprintf(why, why_sz, "missing CACHE_KEY in \"%s\"", path);
        free(key_file);
        return 0;
    }
    if (rA <= 0)
    {
        if (why && why_sz) snprintf(why, why_sz, "missing CACHE_HAS_A in \"%s\"", path);
        free(key_file);
        return 0;
    }

    /* Compute expected key from current sim */
    char *cfg = field_cache_build_config_string(sim, require_A ? 1 : 0);
    if (!cfg)
    {
        if (why && why_sz) snprintf(why, why_sz, "OOM building expected config");
        free(key_file);
        return -2;
    }

    uint64_t h = fnv1a64(cfg, strlen(cfg));
    free(cfg);

    char key_expected[64];
    snprintf(key_expected, sizeof(key_expected), "fnv1a64:0x%016" PRIx64, h);

    if (strcmp(key_file, key_expected) != 0)
    {
        if (why && why_sz) snprintf(why, why_sz, "CACHE_KEY mismatch (file=%s expected=%s)", key_file, key_expected);
        free(key_file);
        return 0;
    }

    if (require_A && !hasA_file)
    {
        if (why && why_sz) snprintf(why, why_sz, "cache does not contain A (CACHE_HAS_A=0)");
        free(key_file);
        return 0;
    }

    free(key_file);
    return 1;
}

/* -------------------------- main entry -------------------------- */

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

    /* Build cache provenance once */
    char *cache_config = field_cache_build_config_string(sim, opt.compute_A);
    if (!cache_config)
    {
        die_root(comm, opt.root_rank, "field_cache: failed to build cache config string (OOM)");
        return 2;
    }

    uint64_t hh = fnv1a64(cache_config, strlen(cache_config));
    char cache_key[64];
    snprintf(cache_key, sizeof(cache_key), "fnv1a64:0x%016" PRIx64, hh);

    if (rank == opt.root_rank)
    {
        int mk = ensure_parent_dir(prefix);
        if (mk != 0)
        {
            fprintf(stderr, "field_cache: failed to create output directory for prefix \"%s\" (rc=%d)\n", prefix, mk);
            free(cache_config);
            MPI_Abort(comm, 100);
        }
    }

    MPI_Barrier(comm);

    const InputGridSpec *g = &sim->grid;

    status_rootf(comm, opt.root_rank,
                 "field_cache: time grid t_min=%f, t_max=%f, dt=%f t_n=%d",
                 g->t_min, g->t_max, g->dt, g->t_n);

    status_rootf(comm, opt.root_rank,
                 "field_cache: spatial grid x1_min=%f, x1_max=%f, dx1=%f, x1_n=%d (axis=%s)",
                 g->ax1_min, g->ax1_max, g->dx1, g->ax1_n, axis_name(g->ax1));

    if (g->has_ax2)
    {
        status_rootf(comm, opt.root_rank,
                     "field_cache: spatial grid x2_min=%f, x2_max=%f, dx2=%f, x2_n=%d (axis=%s)",
                     g->ax2_min, g->ax2_max, g->dx2, g->ax2_n, axis_name(g->ax2));
    }

    status_rootf(comm, opt.root_rank, "field_cache: ranks=%d, output prefix=\"%s\"", nr, prefix);
    status_rootf(comm, opt.root_rank, "field_cache: CACHE_KEY=%s", cache_key);

    int rc = 0;

    if (!g->has_ax2)
    {
        size_t i0 = 0, nloc = 0;
        decompose_1d((size_t)g->ax1_n, rank, nr, &i0, &nloc);

        rc = write_rank_files_2d(g, pulse, prefix, opt.compute_A, i0, nloc, rank,
                                 cache_config, cache_key, opt.compute_A);
        if (rc != 0)
        {
            die_root(comm, opt.root_rank, "field_cache: rank write (2D) failed");
            free(cache_config);
            return 10 + rc;
        }

        MPI_Barrier(comm);

        if (opt.merge_on_root)
        {
            status_root(comm, opt.root_rank, "field_cache: per-rank slabs computed");

            rc = merge_component_2d(g, prefix, "Ex", opt.compute_A, opt.root_rank, comm,
                                    cache_config, cache_key, opt.compute_A);
            if (rc == 0) remove_rank_files(prefix, "Ex", nr, opt.root_rank, comm);
            if (rc != 0) { free(cache_config); return 20 + rc; }

            rc = merge_component_2d(g, prefix, "Ey", opt.compute_A, opt.root_rank, comm,
                                    cache_config, cache_key, opt.compute_A);
            if (rc == 0) remove_rank_files(prefix, "Ey", nr, opt.root_rank, comm);
            if (rc != 0) { free(cache_config); return 21 + rc; }

            rc = merge_component_2d(g, prefix, "Ez", opt.compute_A, opt.root_rank, comm,
                                    cache_config, cache_key, opt.compute_A);
            if (rc == 0) remove_rank_files(prefix, "Ez", nr, opt.root_rank, comm);
            if (rc != 0) { free(cache_config); return 22 + rc; }

            if (opt.compute_A)
            {
                rc = merge_component_2d(g, prefix, "Ax", opt.compute_A, opt.root_rank, comm,
                                        cache_config, cache_key, opt.compute_A);
                if (rc == 0) remove_rank_files(prefix, "Ax", nr, opt.root_rank, comm);
                if (rc != 0) { free(cache_config); return 23 + rc; }

                rc = merge_component_2d(g, prefix, "Ay", opt.compute_A, opt.root_rank, comm,
                                        cache_config, cache_key, opt.compute_A);
                if (rc == 0) remove_rank_files(prefix, "Ay", nr, opt.root_rank, comm);
                if (rc != 0) { free(cache_config); return 24 + rc; }

                rc = merge_component_2d(g, prefix, "Az", opt.compute_A, opt.root_rank, comm,
                                        cache_config, cache_key, opt.compute_A);
                if (rc == 0) remove_rank_files(prefix, "Az", nr, opt.root_rank, comm);
                if (rc != 0) { free(cache_config); return 25 + rc; }
            }

            MPI_Barrier(comm);
        }

        free(cache_config);
        return 0;
    }

    status_root(comm, opt.root_rank, "field_cache: computing per-rank slabs...");

    size_t i0 = 0, nloc = 0;
    decompose_1d((size_t)g->ax2_n, rank, nr, &i0, &nloc);

    rc = write_rank_files_3d(g, pulse, prefix, opt.compute_A, i0, nloc, rank,
                             cache_config, cache_key, opt.compute_A);
    if (rc != 0)
    {
        die_root(comm, opt.root_rank, "field_cache: rank write (3D) failed");
        free(cache_config);
        return 30 + rc;
    }

    MPI_Barrier(comm);
    status_root(comm, opt.root_rank, "field_cache: per-rank slabs computed");

    if (opt.merge_on_root)
    {
        status_root(comm, opt.root_rank, "field_cache: merging rank slabs into final files...");

        rc = merge_component_3d(g, prefix, "Ex", opt.compute_A, opt.root_rank, comm,
                                cache_config, cache_key, opt.compute_A);
        remove_rank_files(prefix, "Ex", nr, opt.root_rank, comm);
        if (rc != 0) { free(cache_config); return 40 + rc; }

        rc = merge_component_3d(g, prefix, "Ey", opt.compute_A, opt.root_rank, comm,
                                cache_config, cache_key, opt.compute_A);
        remove_rank_files(prefix, "Ey", nr, opt.root_rank, comm);
        if (rc != 0) { free(cache_config); return 41 + rc; }

        rc = merge_component_3d(g, prefix, "Ez", opt.compute_A, opt.root_rank, comm,
                                cache_config, cache_key, opt.compute_A);
        remove_rank_files(prefix, "Ez", nr, opt.root_rank, comm);
        if (rc != 0) { free(cache_config); return 42 + rc; }

        if (opt.compute_A)
        {
            rc = merge_component_3d(g, prefix, "Ax", opt.compute_A, opt.root_rank, comm,
                                    cache_config, cache_key, opt.compute_A);
            remove_rank_files(prefix, "Ax", nr, opt.root_rank, comm);
            if (rc != 0) { free(cache_config); return 43 + rc; }

            rc = merge_component_3d(g, prefix, "Ay", opt.compute_A, opt.root_rank, comm,
                                    cache_config, cache_key, opt.compute_A);
            remove_rank_files(prefix, "Ay", nr, opt.root_rank, comm);
            if (rc != 0) { free(cache_config); return 44 + rc; }

            rc = merge_component_3d(g, prefix, "Az", opt.compute_A, opt.root_rank, comm,
                                    cache_config, cache_key, opt.compute_A);
            remove_rank_files(prefix, "Az", nr, opt.root_rank, comm);
            if (rc != 0) { free(cache_config); return 45 + rc; }
        }
    }

    status_root(comm, opt.root_rank, "field_cache: done\n");
    free(cache_config);
    return 0;
}
