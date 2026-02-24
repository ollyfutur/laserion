/* field_cache.c — parallel-HDF5 implementation (no per-rank files, no merge)
 *
 * Behaviour:
 *  - nranks == 1: serial HDF5, write final component files directly.
 *  - nranks  > 1: REQUIRE parallel HDF5; write final component files collectively using MPI-IO.
 *                Each rank writes its own spatial slab hyperslab; time is buffered in blocks.
 *
 * Output naming (unchanged):
 *   <prefix>/Ex.h5, Ey.h5, Ez.h5, (Ax/Ay/Az if compute_A)
 *
 * Notes:
 *  - FieldCacheOptions.merge_on_root is ignored when nranks>1 (kept for API compatibility).
 *  - Uses collective dataset writes (H5FD_MPIO_COLLECTIVE) by default.
 *
 * MODIFICATION (requested):
 *  - Removed HDF5 chunking for cache datasets: datasets are created CONTIGUOUS (no H5Pset_chunk).
 *    Time blocking (t_chunk) is still used only for in-memory buffering.
 *  - We keep a dataset-create property list to disable fill and delay allocation where possible.
 */

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

#ifdef H5_HAVE_PARALLEL
#include <H5FDmpio.h>
#endif

/* -------------------------- timing helpers -------------------------- */

static double now_s(MPI_Comm comm)
{
    (void)comm;
    return MPI_Wtime();
}

/* Gather min/max/avg and print on root */
static void report_time_stats(MPI_Comm comm, int root,
                              const char *label, double t_local)
{
    int rank = 0, nr = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nr);

    if (nr == 1)
    {
        if (rank == root)
        {
            printf("field_cache: timing: %s: %.6f s (serial)\n", label, t_local);
            fflush(stdout);
        }
        return;
    }

    double t_min = 0.0, t_max = 0.0, t_sum = 0.0;
    MPI_Reduce(&t_local, &t_min, 1, MPI_DOUBLE, MPI_MIN, root, comm);
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(&t_local, &t_sum, 1, MPI_DOUBLE, MPI_SUM, root, comm);

    if (rank == root)
    {
        double t_avg = t_sum / (double)nr;
        printf("field_cache: timing: %s: min=%.6f s  avg=%.6f s  max=%.6f s  (nranks=%d)\n",
               label, t_min, t_avg, t_max, nr);
        fflush(stdout);
    }
}

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
    o.merge_on_root = 1; /* kept for API compatibility; ignored in MPI+parallel-HDF5 mode */
    o.root_rank = 0;
    o.io_buffer_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL; /* 4 GiB total budget */
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
    case AXIS_X:
        r_um[0] = a1;
        break;
    case AXIS_Y:
        r_um[1] = a1;
        break;
    case AXIS_Z:
        r_um[2] = a1;
        break;
    default:
        fprintf(stderr, "field_cache: invalid axis in grid spec\n");
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
        default:
            fprintf(stderr, "field_cache: invalid axis in grid spec\n");
            break;
        }
    }
}

static void decompose_1d(size_t n, int rank, int nranks, size_t *i0, size_t *nloc)
{
    size_t base = n / (size_t)nranks;
    size_t rem = n % (size_t)nranks;

    size_t start = (size_t)rank * base + ((rank < (int)rem) ? (size_t)rank : rem);
    size_t count = base + (size_t)((rank < (int)rem) ? 1 : 0);

    *i0 = start;
    *nloc = count;
}

/* -------------------------- attributes (same as before) -------------------------- */

static int h5_write_attr_string(hid_t obj, const char *name, const char *value)
{
    if (!value)
        value = "";

    /* HDF5 fixed-length strings cannot have size 0 */
    size_t n = strlen(value);
    if (n == 0)
        n = 1;

    hid_t atype = H5Tcopy(H5T_C_S1);
    if (atype < 0)
        return 1;

    if (H5Tset_size(atype, n) < 0)
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

/* OSIRIS-style attribute helpers */
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

static void sb_init(StrBuf *s)
{
    s->buf = NULL;
    s->len = 0;
    s->cap = 0;
}
static void sb_free(StrBuf *s)
{
    free(s->buf);
    s->buf = NULL;
    s->len = 0;
    s->cap = 0;
}

static int sb_ensure(StrBuf *s, size_t need_extra)
{
    size_t need = s->len + need_extra + 1;
    if (need <= s->cap)
        return 0;

    size_t newcap = (s->cap == 0) ? 1024 : s->cap;
    while (newcap < need)
        newcap *= 2;

    char *p = (char *)realloc(s->buf, newcap);
    if (!p)
        return 1;

    s->buf = p;
    s->cap = newcap;
    return 0;
}

static int sb_append(StrBuf *s, const char *txt)
{
    if (!txt)
        txt = "";
    size_t n = strlen(txt);
    if (sb_ensure(s, n) != 0)
        return 1;
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
    if (!sim)
        return NULL;

    const InputGridSpec *g = &sim->grid;

    StrBuf s;
    sb_init(&s);

    if (sb_append(&s, "FIELD_CACHE_CONFIG v1\n") != 0)
        goto oom;
    if (sb_appendf(&s, "compute_A=%d\n", compute_A ? 1 : 0) != 0)
        goto oom;

    if (sb_append(&s, "[grid]\n") != 0)
        goto oom;
    if (sb_appendf(&s, "t_min=%.17g\n", g->t_min) != 0)
        goto oom;
    if (sb_appendf(&s, "t_max=%.17g\n", g->t_max) != 0)
        goto oom;
    if (sb_appendf(&s, "dt=%.17g\n", g->dt) != 0)
        goto oom;
    if (sb_appendf(&s, "t_n=%d\n", g->t_n) != 0)
        goto oom;

    if (sb_appendf(&s, "ax1=%s\n", axis_name(g->ax1)) != 0)
        goto oom;
    if (sb_appendf(&s, "has_ax2=%d\n", g->has_ax2 ? 1 : 0) != 0)
        goto oom;
    if (g->has_ax2)
        if (sb_appendf(&s, "ax2=%s\n", axis_name(g->ax2)) != 0)
            goto oom;

    if (sb_appendf(&s, "fixed_x=%.17g\n", g->fixed_x) != 0)
        goto oom;
    if (sb_appendf(&s, "fixed_y=%.17g\n", g->fixed_y) != 0)
        goto oom;
    if (sb_appendf(&s, "fixed_z=%.17g\n", g->fixed_z) != 0)
        goto oom;

    if (sb_appendf(&s, "ax1_min=%.17g\n", g->ax1_min) != 0)
        goto oom;
    if (sb_appendf(&s, "ax1_max=%.17g\n", g->ax1_max) != 0)
        goto oom;
    if (sb_appendf(&s, "dx1=%.17g\n", g->dx1) != 0)
        goto oom;
    if (sb_appendf(&s, "ax1_n=%d\n", g->ax1_n) != 0)
        goto oom;

    if (g->has_ax2)
    {
        if (sb_appendf(&s, "ax2_min=%.17g\n", g->ax2_min) != 0)
            goto oom;
        if (sb_appendf(&s, "ax2_max=%.17g\n", g->ax2_max) != 0)
            goto oom;
        if (sb_appendf(&s, "dx2=%.17g\n", g->dx2) != 0)
            goto oom;
        if (sb_appendf(&s, "ax2_n=%d\n", g->ax2_n) != 0)
            goto oom;
    }

    if (sb_append(&s, "[lasers]\n") != 0)
        goto oom;
    if (sb_appendf(&s, "count=%zu\n", sim->lasers.count) != 0)
        goto oom;

    for (size_t i = 0; i < sim->lasers.count; ++i)
    {
        const InputLaserSpec *L = &sim->lasers.items[i];

        if (sb_appendf(&s, "laser[%zu].type=%d\n", i, (int)L->type) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].E0=%.17g\n", i, L->E0) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].wavelength=%.17g\n", i, L->wavelength) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].phase0=%.17g\n", i, L->phase0) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].k_vec=%.17g,%.17g,%.17g\n",
                       i, L->k_vec[0], L->k_vec[1], L->k_vec[2]) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].r_start=%.17g,%.17g,%.17g\n",
                       i, L->r_start[0], L->r_start[1], L->r_start[2]) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].use_retarded_time=%d\n",
                       i, L->use_retarded_time ? 1 : 0) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].temporal_type=%d\n", i, (int)L->temporal_type) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].tau=%.17g\n", i, L->tau) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].transverse_type=%d\n", i, (int)L->transverse_type) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].w0=%.17g\n", i, L->w0) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].zf=%.17g\n", i, L->zf) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].has_hermite=%d\n", i, L->has_hermite ? 1 : 0) != 0)
            goto oom;
        if (L->has_hermite)
            if (sb_appendf(&s, "laser[%zu].herm_lm=%d,%d\n", i, L->herm_l, L->herm_m) != 0)
                goto oom;

        if (sb_appendf(&s, "laser[%zu].polarization=%d\n", i, (int)L->polarization) != 0)
            goto oom;
        if (sb_appendf(&s, "laser[%zu].angle=%.17g\n", i, L->angle) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].sense=%d\n", i, (int)L->sense) != 0)
            goto oom;

        if (sb_appendf(&s, "laser[%zu].has_jones=%d\n", i, L->has_jones ? 1 : 0) != 0)
            goto oom;
        if (L->has_jones)
        {
            if (sb_appendf(&s, "laser[%zu].p1=%.17g\n", i, L->p1) != 0)
                goto oom;
            if (sb_appendf(&s, "laser[%zu].p2=%.17g\n", i, L->p2) != 0)
                goto oom;
            if (sb_appendf(&s, "laser[%zu].delta=%.17g\n", i, L->delta) != 0)
                goto oom;
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
                               const char *comp)
{
    char fname[128];
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

static const char *axis_long_name_osiris(Axis a)
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

/* -------------------------- buffering policy -------------------------- */

/* Per-rank effective budget:
   - serial: full opt_total
   - MPI: divide total across ranks so total memory stays ~constant
   Also enforce a small floor so we never get chunk=0. */
static uint64_t per_rank_budget_bytes(uint64_t opt_total, int nranks)
{
    const uint64_t floor_bytes = 32ULL * 1024ULL * 1024ULL; /* 32 MiB */
    if (nranks <= 1)
        return (opt_total < floor_bytes) ? floor_bytes : opt_total;

    uint64_t b = opt_total / (uint64_t)nranks;
    if (b < floor_bytes)
        b = floor_bytes;
    return b;
}

/* Compute t-chunk such that ncomp * t_chunk * plane * sizeof(double) <= budget.
   NOTE: This now controls ONLY the in-memory time blocking, not HDF5 chunking. */
static size_t choose_t_chunk(uint64_t budget_bytes, int ncomp, size_t plane, int t_n)
{
    if (plane == 0 || ncomp <= 0)
        return 1;

    uint64_t denom = (uint64_t)ncomp * (uint64_t)plane * (uint64_t)sizeof(double);
    if (denom == 0)
        return 1;

    uint64_t tc = budget_bytes / denom;
    if (tc < 1)
        tc = 1;
    if (tc > (uint64_t)t_n)
        tc = (uint64_t)t_n;

    const uint64_t hard_cap = 16384ULL;
    if (tc > hard_cap)
        tc = hard_cap;

    return (size_t)tc;
}

/* -------------------------- HDF5 file create (serial/parallel) -------------------------- */

static hid_t h5_create_trunc_serial(const char *path)
{
    return H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

static hid_t h5_open_ro_serial(const char *path)
{
    return H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
}

#ifdef H5_HAVE_PARALLEL
static hid_t h5_create_trunc_parallel(const char *path, MPI_Comm comm)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0)
        return -1;

    if (H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) < 0)
    {
        H5Pclose(fapl);
        return -2;
    }

    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    return f;
}
#endif

/* -------------------------- dataset+file creation (single dataset) -------------------------- */

static int create_single_dataset_file(hid_t *out_f, hid_t *out_dset,
                                      const char *path,
                                      const char *component_name,
                                      int nd, const hsize_t *dims,
                                      const InputGridSpec *g,
                                      const char *cache_config,
                                      const char *cache_key,
                                      int cache_has_A,
                                      size_t t_chunk_for_hdf5,
                                      int use_parallel_hdf5,
                                      MPI_Comm comm)
{
    (void)t_chunk_for_hdf5; /* now used only for in-memory blocking, not dataset layout */

    hid_t f = -1;

    if (!use_parallel_hdf5)
    {
        f = h5_create_trunc_serial(path);
    }
    else
    {
#ifdef H5_HAVE_PARALLEL
        f = h5_create_trunc_parallel(path, comm);
#else
        (void)comm;
        return 1001;
#endif
    }

    if (f < 0)
        return 1;

    /* OSIRIS-like attrs + groups (written by all ranks; identical) */
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

    if (cache_config)
        h5_write_attr_string(f, "CACHE_CONFIG", cache_config);
    if (cache_key)
        h5_write_attr_string(f, "CACHE_KEY", cache_key);
    h5_write_attr_int(f, "CACHE_HAS_A", cache_has_A ? 1 : 0);

    /* dataset creation (CONTIGUOUS: no chunking) */
    hid_t space = H5Screate_simple(nd, dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 2;
    }

    char dset_path[64];
    snprintf(dset_path, sizeof(dset_path), "/%s", component_name);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (dcpl >= 0)
    {
        (void)H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
        (void)H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_LATE);
    }

    hid_t dset = H5Dcreate2(f, dset_path, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, (dcpl >= 0 ? dcpl : H5P_DEFAULT), H5P_DEFAULT);

    if (dcpl >= 0)
        H5Pclose(dcpl);
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

/* -------------------------- compatibility check (unchanged; serial open) -------------------------- */

static int h5_read_attr_int(hid_t obj, const char *name, int *out)
{
    if (!out)
        return -1;

    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0)
        return 0;

    herr_t st = H5Aread(a, H5T_NATIVE_INT, out);
    H5Aclose(a);

    return (st < 0) ? -2 : 1;
}

static int h5_read_attr_string_alloc(hid_t obj, const char *name, char **out_s)
{
    if (!out_s)
        return -1;
    *out_s = NULL;

    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0)
        return 0;

    hid_t t = H5Aget_type(a);
    if (t < 0)
    {
        H5Aclose(a);
        return -2;
    }

    size_t sz = (size_t)H5Tget_size(t);
    if (sz == 0)
    {
        H5Tclose(t);
        H5Aclose(a);
        return -3;
    }

    char *buf = (char *)malloc(sz + 1);
    if (!buf)
    {
        H5Tclose(t);
        H5Aclose(a);
        return -4;
    }

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

int field_cache_is_compatible(const char *cache_dir, const InputSimSpec *sim, int require_A,
                              char *why, size_t why_sz)
{
    if (why && why_sz)
        why[0] = '\0';
    if (!cache_dir || !sim)
        return -1;

    char path[512];
    component_filename(path, sizeof(path), cache_dir, "Ex");

    hid_t f = h5_open_ro_serial(path);
    if (f < 0)
    {
        if (why && why_sz)
            snprintf(why, why_sz, "cannot open \"%s\"", path);
        return 0;
    }

    char *key_file = NULL;
    int hasA_file = 0;

    int rkey = h5_read_attr_string_alloc(f, "CACHE_KEY", &key_file);
    int rA = h5_read_attr_int(f, "CACHE_HAS_A", &hasA_file);

    H5Fclose(f);

    if (rkey <= 0 || !key_file)
    {
        if (why && why_sz)
            snprintf(why, why_sz, "missing CACHE_KEY in \"%s\"", path);
        free(key_file);
        return 0;
    }
    if (rA <= 0)
    {
        if (why && why_sz)
            snprintf(why, why_sz, "missing CACHE_HAS_A in \"%s\"", path);
        free(key_file);
        return 0;
    }

    char *cfg = field_cache_build_config_string(sim, require_A ? 1 : 0);
    if (!cfg)
    {
        if (why && why_sz)
            snprintf(why, why_sz, "OOM building expected config");
        free(key_file);
        return -2;
    }

    uint64_t h = fnv1a64(cfg, strlen(cfg));
    free(cfg);

    char key_expected[64];
    snprintf(key_expected, sizeof(key_expected), "fnv1a64:0x%016" PRIx64, h);

    if (strcmp(key_file, key_expected) != 0)
    {
        if (why && why_sz)
            snprintf(why, why_sz, "CACHE_KEY mismatch (file=%s expected=%s)", key_file, key_expected);
        free(key_file);
        return 0;
    }

    if (require_A && !hasA_file)
    {
        if (why && why_sz)
            snprintf(why, why_sz, "cache does not contain A (CACHE_HAS_A=0)");
        free(key_file);
        return 0;
    }

    free(key_file);
    return 1;
}

/* -------------------------- parallel slab writers (buffered time blocks) -------------------------- */

static int write_parallel_files_2d(const InputGridSpec *g,
                                   const LaserPulse *pulse,
                                   const char *prefix,
                                   int compute_A,
                                   size_t a1_i0, size_t a1_nloc,
                                   const char *cache_config,
                                   const char *cache_key,
                                   int cache_has_A,
                                   size_t t_chunk,
                                   int use_parallel_hdf5,
                                   MPI_Comm comm)
{
    /* global dims: [t, ax1] */
    hsize_t gdims[2] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n};

    hid_t fEx = -1, dEx = -1, fEy = -1, dEy = -1, fEz = -1, dEz = -1;
    hid_t fAx = -1, dAx = -1, fAy = -1, dAy = -1, fAz = -1, dAz = -1;

    char path[512];

    component_filename(path, sizeof(path), prefix, "Ex");
    if (create_single_dataset_file(&fEx, &dEx, path, "Ex", 2, gdims, g, cache_config, cache_key, cache_has_A,
                                   t_chunk, use_parallel_hdf5, comm) != 0)
        return 1;

    component_filename(path, sizeof(path), prefix, "Ey");
    if (create_single_dataset_file(&fEy, &dEy, path, "Ey", 2, gdims, g, cache_config, cache_key, cache_has_A,
                                   t_chunk, use_parallel_hdf5, comm) != 0)
        return 2;

    component_filename(path, sizeof(path), prefix, "Ez");
    if (create_single_dataset_file(&fEz, &dEz, path, "Ez", 2, gdims, g, cache_config, cache_key, cache_has_A,
                                   t_chunk, use_parallel_hdf5, comm) != 0)
        return 3;

    if (compute_A)
    {
        component_filename(path, sizeof(path), prefix, "Ax");
        if (create_single_dataset_file(&fAx, &dAx, path, "Ax", 2, gdims, g, cache_config, cache_key, cache_has_A,
                                       t_chunk, use_parallel_hdf5, comm) != 0)
            return 4;

        component_filename(path, sizeof(path), prefix, "Ay");
        if (create_single_dataset_file(&fAy, &dAy, path, "Ay", 2, gdims, g, cache_config, cache_key, cache_has_A,
                                       t_chunk, use_parallel_hdf5, comm) != 0)
            return 5;

        component_filename(path, sizeof(path), prefix, "Az");
        if (create_single_dataset_file(&fAz, &dAz, path, "Az", 2, gdims, g, cache_config, cache_key, cache_has_A,
                                       t_chunk, use_parallel_hdf5, comm) != 0)
            return 6;
    }

    /* dxpl for collective writes in parallel mode */
    hid_t dxpl = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
    hid_t dxpl_local = -1;
    if (use_parallel_hdf5)
    {
        dxpl_local = H5Pcreate(H5P_DATASET_XFER);
        if (dxpl_local >= 0)
            (void)H5Pset_dxpl_mpio(dxpl_local, H5FD_MPIO_COLLECTIVE);
        dxpl = (dxpl_local >= 0) ? dxpl_local : H5P_DEFAULT;
    }
#endif

    /* per-time scratch */
    double *Ex_line = (double *)malloc(a1_nloc * sizeof(double));
    double *Ey_line = (double *)malloc(a1_nloc * sizeof(double));
    double *Ez_line = (double *)malloc(a1_nloc * sizeof(double));
    double *Ax_line = compute_A ? (double *)malloc(a1_nloc * sizeof(double)) : NULL;
    double *Ay_line = compute_A ? (double *)malloc(a1_nloc * sizeof(double)) : NULL;
    double *Az_line = compute_A ? (double *)malloc(a1_nloc * sizeof(double)) : NULL;
    double *Ex_prev = compute_A ? (double *)malloc(a1_nloc * sizeof(double)) : NULL;
    double *Ey_prev = compute_A ? (double *)malloc(a1_nloc * sizeof(double)) : NULL;
    double *Ez_prev = compute_A ? (double *)malloc(a1_nloc * sizeof(double)) : NULL;

    if (!Ex_line || !Ey_line || !Ez_line ||
        (compute_A && (!Ax_line || !Ay_line || !Az_line || !Ex_prev || !Ey_prev || !Ez_prev)))
        return 20;

    size_t block_cap = (t_chunk < 1) ? 1 : t_chunk;

    double *Ex_blk = (double *)malloc(block_cap * a1_nloc * sizeof(double));
    double *Ey_blk = (double *)malloc(block_cap * a1_nloc * sizeof(double));
    double *Ez_blk = (double *)malloc(block_cap * a1_nloc * sizeof(double));
    double *Ax_blk = compute_A ? (double *)malloc(block_cap * a1_nloc * sizeof(double)) : NULL;
    double *Ay_blk = compute_A ? (double *)malloc(block_cap * a1_nloc * sizeof(double)) : NULL;
    double *Az_blk = compute_A ? (double *)malloc(block_cap * a1_nloc * sizeof(double)) : NULL;

    if (!Ex_blk || !Ey_blk || !Ez_blk || (compute_A && (!Ax_blk || !Ay_blk || !Az_blk)))
        return 21;

    int t0 = 0;
    int nb = 0;

    for (int it = 0; it < g->t_n; ++it)
    {
        const double t = g->t_min + (double)it * g->dt;

        for (size_t ia = 0; ia < a1_nloc; ++ia)
        {
            const size_t ig = a1_i0 + ia;
            const double a1 = g->ax1_min + (double)ig * g->dx1;

            double r[3];
            set_r_from_axes(g, a1, 0.0, r);

            double E[3];
            LaserPulse_E(pulse, t, r, E);

            Ex_line[ia] = E[0];
            Ey_line[ia] = E[1];
            Ez_line[ia] = E[2];
        }

        if (compute_A)
        {
            if (it == 0)
            {
                for (size_t ia = 0; ia < a1_nloc; ++ia)
                {
                    Ax_line[ia] = Ay_line[ia] = Az_line[ia] = 0.0;
                    Ex_prev[ia] = Ex_line[ia];
                    Ey_prev[ia] = Ey_line[ia];
                    Ez_prev[ia] = Ez_line[ia];
                }
            }
            else
            {
                const double dt = g->dt;
                for (size_t ia = 0; ia < a1_nloc; ++ia)
                {
                    Ax_line[ia] -= 0.5 * (Ex_prev[ia] + Ex_line[ia]) * dt;
                    Ay_line[ia] -= 0.5 * (Ey_prev[ia] + Ey_line[ia]) * dt;
                    Az_line[ia] -= 0.5 * (Ez_prev[ia] + Ez_line[ia]) * dt;

                    Ex_prev[ia] = Ex_line[ia];
                    Ey_prev[ia] = Ey_line[ia];
                    Ez_prev[ia] = Ez_line[ia];
                }
            }
        }

        memcpy(Ex_blk + (size_t)nb * a1_nloc, Ex_line, a1_nloc * sizeof(double));
        memcpy(Ey_blk + (size_t)nb * a1_nloc, Ey_line, a1_nloc * sizeof(double));
        memcpy(Ez_blk + (size_t)nb * a1_nloc, Ez_line, a1_nloc * sizeof(double));
        if (compute_A)
        {
            memcpy(Ax_blk + (size_t)nb * a1_nloc, Ax_line, a1_nloc * sizeof(double));
            memcpy(Ay_blk + (size_t)nb * a1_nloc, Ay_line, a1_nloc * sizeof(double));
            memcpy(Az_blk + (size_t)nb * a1_nloc, Az_line, a1_nloc * sizeof(double));
        }
        nb++;

        if ((size_t)nb == block_cap || it == g->t_n - 1)
        {
            /* file hyperslab: [t0: t0+nb, a1_i0 : a1_i0+a1_nloc] */
            hsize_t start[2] = {(hsize_t)t0, (hsize_t)a1_i0};
            hsize_t count[2] = {(hsize_t)nb, (hsize_t)a1_nloc};

            hid_t mspace = H5Screate_simple(2, count, NULL);
            if (mspace < 0)
                return 30;

#define WRITE_ONE_2D(DSET, BUF)                                                        \
    do                                                                                 \
    {                                                                                  \
        hid_t fspace = H5Dget_space((DSET));                                           \
        if (fspace < 0)                                                                \
        {                                                                              \
            H5Sclose(mspace);                                                          \
            return 31;                                                                 \
        }                                                                              \
        if (H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL) < 0) \
        {                                                                              \
            H5Sclose(fspace);                                                          \
            H5Sclose(mspace);                                                          \
            return 32;                                                                 \
        }                                                                              \
        if (H5Dwrite((DSET), H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, (BUF)) < 0)      \
        {                                                                              \
            H5Sclose(fspace);                                                          \
            H5Sclose(mspace);                                                          \
            return 33;                                                                 \
        }                                                                              \
        H5Sclose(fspace);                                                              \
    } while (0)

            WRITE_ONE_2D(dEx, Ex_blk);
            WRITE_ONE_2D(dEy, Ey_blk);
            WRITE_ONE_2D(dEz, Ez_blk);
            if (compute_A)
            {
                WRITE_ONE_2D(dAx, Ax_blk);
                WRITE_ONE_2D(dAy, Ay_blk);
                WRITE_ONE_2D(dAz, Az_blk);
            }

#undef WRITE_ONE_2D

            H5Sclose(mspace);

            t0 += nb;
            nb = 0;
        }
    }

    free(Ex_line);
    free(Ey_line);
    free(Ez_line);
    free(Ax_line);
    free(Ay_line);
    free(Az_line);
    free(Ex_prev);
    free(Ey_prev);
    free(Ez_prev);

    free(Ex_blk);
    free(Ey_blk);
    free(Ez_blk);
    free(Ax_blk);
    free(Ay_blk);
    free(Az_blk);

#ifdef H5_HAVE_PARALLEL
    if (use_parallel_hdf5 && dxpl_local >= 0)
        H5Pclose(dxpl_local);
#endif

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

static int write_parallel_files_3d(const InputGridSpec *g,
                                   const LaserPulse *pulse,
                                   const char *prefix,
                                   int compute_A,
                                   size_t a1_i0, size_t a1_nloc, /* <-- decompose ax1 now */
                                   const char *cache_config,
                                   const char *cache_key,
                                   int cache_has_A,
                                   size_t t_chunk,
                                   int use_parallel_hdf5,
                                   MPI_Comm comm)
{
    /* global dims: [t, ax1, ax2] ; decompose ax1 (middle dim) */
    hsize_t gdims[3] = {(hsize_t)g->t_n, (hsize_t)g->ax1_n, (hsize_t)g->ax2_n};

    hid_t fEx = -1, dEx = -1, fEy = -1, dEy = -1, fEz = -1, dEz = -1;
    hid_t fAx = -1, dAx = -1, fAy = -1, dAy = -1, fAz = -1, dAz = -1;

    char path[512];

    component_filename(path, sizeof(path), prefix, "Ex");
    if (create_single_dataset_file(&fEx, &dEx, path, "Ex", 3, gdims, g, cache_config, cache_key, cache_has_A,
                                   t_chunk, use_parallel_hdf5, comm) != 0)
        return 1;

    component_filename(path, sizeof(path), prefix, "Ey");
    if (create_single_dataset_file(&fEy, &dEy, path, "Ey", 3, gdims, g, cache_config, cache_key, cache_has_A,
                                   t_chunk, use_parallel_hdf5, comm) != 0)
        return 2;

    component_filename(path, sizeof(path), prefix, "Ez");
    if (create_single_dataset_file(&fEz, &dEz, path, "Ez", 3, gdims, g, cache_config, cache_key, cache_has_A,
                                   t_chunk, use_parallel_hdf5, comm) != 0)
        return 3;

    if (compute_A)
    {
        component_filename(path, sizeof(path), prefix, "Ax");
        if (create_single_dataset_file(&fAx, &dAx, path, "Ax", 3, gdims, g, cache_config, cache_key, cache_has_A,
                                       t_chunk, use_parallel_hdf5, comm) != 0)
            return 4;

        component_filename(path, sizeof(path), prefix, "Ay");
        if (create_single_dataset_file(&fAy, &dAy, path, "Ay", 3, gdims, g, cache_config, cache_key, cache_has_A,
                                       t_chunk, use_parallel_hdf5, comm) != 0)
            return 5;

        component_filename(path, sizeof(path), prefix, "Az");
        if (create_single_dataset_file(&fAz, &dAz, path, "Az", 3, gdims, g, cache_config, cache_key, cache_has_A,
                                       t_chunk, use_parallel_hdf5, comm) != 0)
            return 6;
    }

    /* collective dxpl */
    hid_t dxpl = H5P_DEFAULT;
#ifdef H5_HAVE_PARALLEL
    hid_t dxpl_local = -1;
    if (use_parallel_hdf5)
    {
        dxpl_local = H5Pcreate(H5P_DATASET_XFER);
        if (dxpl_local >= 0)
            (void)H5Pset_dxpl_mpio(dxpl_local, H5FD_MPIO_COLLECTIVE);
        dxpl = (dxpl_local >= 0) ? dxpl_local : H5P_DEFAULT;
    }
#endif

    /* local plane is contiguous in ax2 */
    const size_t plane = a1_nloc * (size_t)g->ax2_n;

    double *Ex = (double *)malloc(plane * sizeof(double));
    double *Ey = (double *)malloc(plane * sizeof(double));
    double *Ez = (double *)malloc(plane * sizeof(double));
    double *Ax = compute_A ? (double *)malloc(plane * sizeof(double)) : NULL;
    double *Ay = compute_A ? (double *)malloc(plane * sizeof(double)) : NULL;
    double *Az = compute_A ? (double *)malloc(plane * sizeof(double)) : NULL;
    double *Ex_prev = compute_A ? (double *)malloc(plane * sizeof(double)) : NULL;
    double *Ey_prev = compute_A ? (double *)malloc(plane * sizeof(double)) : NULL;
    double *Ez_prev = compute_A ? (double *)malloc(plane * sizeof(double)) : NULL;

    if (!Ex || !Ey || !Ez || (compute_A && (!Ax || !Ay || !Az || !Ex_prev || !Ey_prev || !Ez_prev)))
        return 20;

    size_t block_cap = (t_chunk < 1) ? 1 : t_chunk;

    double *Ex_blk = (double *)malloc(block_cap * plane * sizeof(double));
    double *Ey_blk = (double *)malloc(block_cap * plane * sizeof(double));
    double *Ez_blk = (double *)malloc(block_cap * plane * sizeof(double));
    double *Ax_blk = compute_A ? (double *)malloc(block_cap * plane * sizeof(double)) : NULL;
    double *Ay_blk = compute_A ? (double *)malloc(block_cap * plane * sizeof(double)) : NULL;
    double *Az_blk = compute_A ? (double *)malloc(block_cap * plane * sizeof(double)) : NULL;

    if (!Ex_blk || !Ey_blk || !Ez_blk || (compute_A && (!Ax_blk || !Ay_blk || !Az_blk)))
        return 21;

    int t0 = 0;
    int nb = 0;

    for (int it = 0; it < g->t_n; ++it)
    {
        const double t = g->t_min + (double)it * g->dt;

        /* fill local slab */
        for (size_t i1loc = 0; i1loc < a1_nloc; ++i1loc)
        {
            const size_t i1g = a1_i0 + i1loc;
            const double a1 = g->ax1_min + (double)i1g * g->dx1;

            for (int i2 = 0; i2 < g->ax2_n; ++i2)
            {
                const double a2 = g->ax2_min + (double)i2 * g->dx2;

                double r[3];
                set_r_from_axes(g, a1, a2, r);

                double E[3];
                LaserPulse_E(pulse, t, r, E);

                const size_t idx = i1loc * (size_t)g->ax2_n + (size_t)i2;
                Ex[idx] = E[0];
                Ey[idx] = E[1];
                Ez[idx] = E[2];
            }
        }

        if (compute_A)
        {
            if (it == 0)
            {
                for (size_t k = 0; k < plane; ++k)
                {
                    Ax[k] = Ay[k] = Az[k] = 0.0;
                    Ex_prev[k] = Ex[k];
                    Ey_prev[k] = Ey[k];
                    Ez_prev[k] = Ez[k];
                }
            }
            else
            {
                const double dt = g->dt;
                for (size_t k = 0; k < plane; ++k)
                {
                    Ax[k] -= 0.5 * (Ex_prev[k] + Ex[k]) * dt;
                    Ay[k] -= 0.5 * (Ey_prev[k] + Ey[k]) * dt;
                    Az[k] -= 0.5 * (Ez_prev[k] + Ez[k]) * dt;

                    Ex_prev[k] = Ex[k];
                    Ey_prev[k] = Ey[k];
                    Ez_prev[k] = Ez[k];
                }
            }
        }

        memcpy(Ex_blk + (size_t)nb * plane, Ex, plane * sizeof(double));
        memcpy(Ey_blk + (size_t)nb * plane, Ey, plane * sizeof(double));
        memcpy(Ez_blk + (size_t)nb * plane, Ez, plane * sizeof(double));
        if (compute_A)
        {
            memcpy(Ax_blk + (size_t)nb * plane, Ax, plane * sizeof(double));
            memcpy(Ay_blk + (size_t)nb * plane, Ay, plane * sizeof(double));
            memcpy(Az_blk + (size_t)nb * plane, Az, plane * sizeof(double));
        }
        nb++;

        if ((size_t)nb == block_cap || it == g->t_n - 1)
        {
            /* contiguous hyperslab for each rank: [t0:t0+nb, a1_i0:a1_i0+a1_nloc, 0:ax2_n] */
            hsize_t start[3] = {(hsize_t)t0, (hsize_t)a1_i0, 0};
            hsize_t count[3] = {(hsize_t)nb, (hsize_t)a1_nloc, (hsize_t)g->ax2_n};

            hid_t mspace = H5Screate_simple(3, count, NULL);
            if (mspace < 0)
                return 30;

#define WRITE_ONE_3D(DSET, BUF)                                                        \
    do                                                                                 \
    {                                                                                  \
        hid_t fspace = H5Dget_space((DSET));                                           \
        if (fspace < 0)                                                                \
        {                                                                              \
            H5Sclose(mspace);                                                          \
            return 31;                                                                 \
        }                                                                              \
        if (H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL) < 0) \
        {                                                                              \
            H5Sclose(fspace);                                                          \
            H5Sclose(mspace);                                                          \
            return 32;                                                                 \
        }                                                                              \
        if (H5Dwrite((DSET), H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, (BUF)) < 0)      \
        {                                                                              \
            H5Sclose(fspace);                                                          \
            H5Sclose(mspace);                                                          \
            return 33;                                                                 \
        }                                                                              \
        H5Sclose(fspace);                                                              \
    } while (0)

            WRITE_ONE_3D(dEx, Ex_blk);
            WRITE_ONE_3D(dEy, Ey_blk);
            WRITE_ONE_3D(dEz, Ez_blk);
            if (compute_A)
            {
                WRITE_ONE_3D(dAx, Ax_blk);
                WRITE_ONE_3D(dAy, Ay_blk);
                WRITE_ONE_3D(dAz, Az_blk);
            }

#undef WRITE_ONE_3D

            H5Sclose(mspace);

            t0 += nb;
            nb = 0;
        }
    }

    free(Ex);
    free(Ey);
    free(Ez);
    free(Ax);
    free(Ay);
    free(Az);
    free(Ex_prev);
    free(Ey_prev);
    free(Ez_prev);

    free(Ex_blk);
    free(Ey_blk);
    free(Ez_blk);
    free(Ax_blk);
    free(Ay_blk);
    free(Az_blk);

#ifdef H5_HAVE_PARALLEL
    if (use_parallel_hdf5 && dxpl_local >= 0)
        H5Pclose(dxpl_local);
#endif

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

    /* Enforce: MPI runs require parallel HDF5 */
    int use_parallel_hdf5 = (nr > 1) ? 1 : 0;
    if (nr > 1)
    {
#ifndef H5_HAVE_PARALLEL
        if (rank == opt.root_rank)
        {
            fprintf(stderr,
                    "field_cache: error: MPI run (nranks=%d) requires Parallel HDF5 (H5_HAVE_PARALLEL not defined).\n"
                    "field_cache: rebuild/link against an MPI-enabled HDF5 and recompile.\n",
                    nr);
        }
        free(cache_config);
        return 500;
#else
        if (rank == opt.root_rank)
            status_root(comm, opt.root_rank, "field_cache: MPI mode — using Parallel HDF5 (MPI-IO), no merge step");
#endif
    }
    else
    {
        status_root(comm, opt.root_rank, "field_cache: serial mode — using standard HDF5");
    }

    uint64_t budget_rank = per_rank_budget_bytes(opt.io_buffer_bytes, nr);
    if (rank == opt.root_rank)
    {
        status_rootf(comm, opt.root_rank,
                     "field_cache: io_buffer_bytes(total)=%" PRIu64 "  => per-rank budget=%" PRIu64 " bytes",
                     opt.io_buffer_bytes, budget_rank);
    }

    const int ncomp_write = opt.compute_A ? 6 : 3;

    int rc = 0;

    if (!g->has_ax2)
    {
        size_t i0 = 0, nloc = 0;
        if (nr == 1)
        {
            i0 = 0;
            nloc = (size_t)g->ax1_n;
        }
        else
        {
            decompose_1d((size_t)g->ax1_n, rank, nr, &i0, &nloc);
        }

        size_t plane = nloc;
        size_t t_chunk = choose_t_chunk(budget_rank, ncomp_write, plane, g->t_n);

        MPI_Barrier(comm);
        double t0 = now_s(comm);

        rc = write_parallel_files_2d(g, pulse, prefix, opt.compute_A, i0, nloc,
                                     cache_config, cache_key, opt.compute_A, t_chunk,
                                     use_parallel_hdf5, comm);

        double t1 = now_s(comm);
        report_time_stats(comm, opt.root_rank, "write cache (2D)", t1 - t0);
    }
    else
    {
        size_t i0 = 0, nloc = 0;
        if (nr == 1)
        {
            i0 = 0;
            nloc = (size_t)g->ax1_n;
        }
        else
        {
            /* decompose ax1 (middle dim) for contiguous writes */
            decompose_1d((size_t)g->ax1_n, rank, nr, &i0, &nloc);
        }

        size_t plane = nloc * (size_t)g->ax2_n;
        size_t t_chunk = choose_t_chunk(budget_rank, ncomp_write, plane, g->t_n);

        MPI_Barrier(comm);
        double t0s = now_s(comm);

        rc = write_parallel_files_3d(g, pulse, prefix, opt.compute_A, i0, nloc,
                                     cache_config, cache_key, opt.compute_A, t_chunk,
                                     use_parallel_hdf5, comm);

        double t1s = now_s(comm);
        report_time_stats(comm, opt.root_rank, "write cache (3D)", t1s - t0s);
    }

    if (rc != 0)
    {
        die_root(comm, opt.root_rank, "field_cache: write failed");
        free(cache_config);
        return 600 + rc;
    }

    status_root(comm, opt.root_rank, "field_cache: done");
    free(cache_config);
    return 0;
}
