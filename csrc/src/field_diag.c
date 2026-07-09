#include "field_diag.h"

#include "diag_h5.h"

#include <hdf5.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------ defaults ---------------------------------- */

FieldDiagOptions field_diag_default_options(void)
{
    FieldDiagOptions o;
    o.root_rank = 0;
    o.write_time_iter_0 = 1;
    return o;
}

/* ------------------------------ utilities --------------------------------- */

static void join_path(char *out, size_t outsz, const char *a, const char *b)
{
    if (!a)
        a = "";
    if (!b)
        b = "";
    const size_t na = strlen(a);
    const int need_slash = (na > 0 && a[na - 1] != '/');
    if (need_slash)
        snprintf(out, outsz, "%s/%s", a, b);
    else
        snprintf(out, outsz, "%s%s", a, b);
}

static int axis_char_to_diag_id(char c, DiagAxisID *out)
{
    c = (char)tolower((unsigned char)c);
    switch (c)
    {
    case 't':
        *out = DIAG_AXIS_T;
        return 0;
    case 'x':
        *out = DIAG_AXIS_X;
        return 0;
    case 'y':
        *out = DIAG_AXIS_Y;
        return 0;
    case 'z':
        *out = DIAG_AXIS_Z;
        return 0;
    default:
        return 1;
    }
}

static int axis_char_to_grid_axis(char c, Axis *out)
{
    c = (char)tolower((unsigned char)c);
    switch (c)
    {
    case 'x':
        *out = AXIS_X;
        return 0;
    case 'y':
        *out = AXIS_Y;
        return 0;
    case 'z':
        *out = AXIS_Z;
        return 0;
    default:
        return 1;
    }
}

static const char *axis_units_for_char(char c)
{
    c = (char)tolower((unsigned char)c);
    if (c == 't')
        return "fs";
    if (c == 'x' || c == 'y' || c == 'z')
        return "\\mu m";
    return "";
}

static const char *axis_long_name_for_char(char c)
{
    c = (char)tolower((unsigned char)c);
    if (c == 't')
        return "t";
    if (c == 'x')
        return "x";
    if (c == 'y')
        return "y";
    if (c == 'z')
        return "z";
    return "";
}

static double get_fixed_pos_for_axis_char(const FieldDiagSpec *d, char c)
{
    c = (char)tolower((unsigned char)c);
    switch (c)
    {
    case 't':
        return d->pos_t;
    case 'x':
        return d->pos_x;
    case 'y':
        return d->pos_y;
    case 'z':
        return d->pos_z;
    default:
        return 0.0;
    }
}

static void fill_fixed_coords_from_spec(DiagFixedCoords *f, const FieldDiagSpec *d)
{
    f->t = d->pos_t;
    f->x = d->pos_x;
    f->y = d->pos_y;
    f->z = d->pos_z;
}

/* Nearest index on uniform grid, clamped to [0, n-1]. */
static size_t nearest_index_uniform(double v, double vmin, double dv, size_t n)
{
    if (n == 0)
        return 0;
    if (!(dv > 0.0))
        return 0;

    double x = (v - vmin) / dv;
    long idx = (long)llround(x);
    if (idx < 0)
        idx = 0;
    if ((size_t)idx >= n)
        idx = (long)(n - 1);
    return (size_t)idx;
}

static void default_units_label_for_comp(const char *comp,
                                         char *units, size_t units_sz,
                                         char *label, size_t label_sz)
{
    if (units && units_sz)
        units[0] = '\0';
    if (label && label_sz)
        label[0] = '\0';
    if (!comp)
        return;

    /* component is "Ex","Ey","Ez","Ax","Ay","Az" */
    const char C0 = comp[0];
    const char C1 = comp[1];

    if (units && units_sz)
    {
        if (C0 == 'E')
            snprintf(units, units_sz, "m_e c \\omega_p e^{-1}");
        else if (C0 == 'A')
            snprintf(units, units_sz, "m_e c^2 e^{-1}");
    }

    if (label && label_sz)
    {
        if (C0 == 'E')
            snprintf(label, label_sz, "E_%c", C1);
        else if (C0 == 'A')
            snprintf(label, label_sz, "A_%c", C1);
        else
            snprintf(label, label_sz, "%s", comp);
    }
}

static int axes_contains(const char *axes, char c)
{
    if (!axes)
        return 0;
    c = (char)tolower((unsigned char)c);
    for (const char *p = axes; *p; ++p)
        if ((char)tolower((unsigned char)*p) == c)
            return 1;
    return 0;
}

static char grid_axis_to_char(Axis a)
{
    switch (a)
    {
    case AXIS_X:
        return 'x';
    case AXIS_Y:
        return 'y';
    case AXIS_Z:
        return 'z';
    default:
        return 'x';
    }
}

static double get_pos_for_axis_char(const FieldDiagSpec *d, char c)
{
    c = (char)tolower((unsigned char)c);
    switch (c)
    {
    case 't':
        return d->pos_t;
    case 'x':
        return d->pos_x;
    case 'y':
        return d->pos_y;
    case 'z':
        return d->pos_z;
    default:
        return 0.0;
    }
}

/* Build "diag/Ex_vs_z__t_10__x_5.h5" style names into out_path. */
static void build_field_diag_path(char *out_path, size_t out_sz,
                                  const char *prefix, /* e.g. "diag" or "diag/fields" */
                                  const char *comp,   /* "Ex" */
                                  const char *axes,   /* "t", "z", "tz", ... */
                                  const InputGridSpec *g,
                                  const FieldDiagSpec *d)
{
    /* Determine which cache axes exist. Cache dims are: t + ax1 (+ ax2). */
    const char ax1c = grid_axis_to_char(g->ax1);
    const char ax2c = grid_axis_to_char(g->ax2);

    /* Compute indices for the cache axes (even if varying; we’ll only print fixed ones). */
    const size_t it = nearest_index_uniform(d->pos_t, g->t_min, g->dt, (size_t)g->t_n);
    const size_t i1 = nearest_index_uniform(get_pos_for_axis_char(d, ax1c),
                                            g->ax1_min, g->dx1, (size_t)g->ax1_n);
    const size_t i2 = g->has_ax2
                          ? nearest_index_uniform(get_pos_for_axis_char(d, ax2c),
                                                  g->ax2_min, g->dx2, (size_t)g->ax2_n)
                          : 0;

    /* Start with "<prefix>/<comp>_vs_<axes>".
       If you pass prefix="diag/fields", you get diag/fieldsEx_... (bad),
       so prefer prefix as a directory ("diag") and we write inside it.
     */
    char base[512];
    snprintf(base, sizeof(base), "%s/%s_vs_%s", prefix, comp, axes);

    /* Append fixed-axis slice numbers (only those not in varying axes). */
    char suffix[512];
    suffix[0] = '\0';

    if (!axes_contains(axes, 't'))
        snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), "__t_%zu", it);

    if (!axes_contains(axes, ax1c))
        snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), "__%c_%zu", ax1c, i1);

    if (g->has_ax2 && !axes_contains(axes, ax2c))
        snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), "__%c_%zu", ax2c, i2);

    snprintf(out_path, out_sz, "%s%s.h5", base, suffix);
}

/* -------------------------- cache layout inference ------------------------ */

/*
 * Infer file dimension semantics by matching sizes to {t_n, ax1_n, ax2_n}.
 * rank2: must match t and ax1.
 * rank3: must match t, ax1, ax2 (and sim must have ax2).
 */
static int infer_dim_semantics(int rank,
                               const hsize_t *dims,
                               size_t t_n,
                               size_t ax1_n,
                               size_t ax2_n,
                               int has_ax2,
                               char *dim_sem /* length rank */)
{
    if (!dims || !dim_sem)
        return 1;
    if (!(rank == 2 || rank == 3))
        return 2;

    for (int k = 0; k < rank; ++k)
        dim_sem[k] = '?';

    int used_t = 0, used_1 = 0, used_2 = 0;

    /* assign t */
    for (int k = 0; k < rank; ++k)
    {
        if (!used_t && (size_t)dims[k] == t_n)
        {
            dim_sem[k] = 't';
            used_t = 1;
        }
    }
    /* assign ax1 */
    for (int k = 0; k < rank; ++k)
    {
        if (dim_sem[k] != '?')
            continue;
        if (!used_1 && (size_t)dims[k] == ax1_n)
        {
            dim_sem[k] = '1';
            used_1 = 1;
        }
    }
    /* assign ax2 */
    for (int k = 0; k < rank; ++k)
    {
        if (dim_sem[k] != '?')
            continue;
        if (rank == 3 && has_ax2 && !used_2 && (size_t)dims[k] == ax2_n)
        {
            dim_sem[k] = '2';
            used_2 = 1;
        }
    }

    for (int k = 0; k < rank; ++k)
        if (dim_sem[k] == '?')
            return 3;

    if (!used_t)
        return 4;
    if (!used_1)
        return 5;
    if (rank == 3)
    {
        if (!has_ax2)
            return 6;
        if (!used_2)
            return 7;
    }
    return 0;
}

static int sem_to_k(char sem, const char *dim_sem, int rank, int *out_k)
{
    for (int k = 0; k < rank; ++k)
    {
        if (dim_sem[k] == sem)
        {
            if (out_k)
                *out_k = k;
            return 0;
        }
    }
    return 1;
}

/* -------------------------- indexing helpers ----------------------------- */

/* Set semantic indices (it,i1,i2) for a given axis char and value index. */
static void set_sem_indices_for_axis(char axc, size_t val,
                                     Axis sim_ax1, Axis sim_ax2, int has_ax2,
                                     size_t *it, size_t *i1, size_t *i2)
{
    axc = (char)tolower((unsigned char)axc);
    if (axc == 't')
    {
        *it = val;
        return;
    }

    Axis asp;
    if (axis_char_to_grid_axis(axc, &asp) != 0)
        return;

    if (asp == sim_ax1)
        *i1 = val;
    else if (has_ax2 && asp == sim_ax2)
        *i2 = val;
}

static int value_in_range(double v, double vmin, double vmax, double dv)
{
    /* accept a tiny tolerance; also allow half-cell beyond endpoints */
    const double tol = 1e-12;
    const double pad = 0.5 * (dv > 0.0 ? dv : 0.0);
    return (v >= (vmin - pad - tol)) && (v <= (vmax + pad + tol));
}

static int check_fixed_axis_in_cache(const InputGridSpec *g,
                                     char axis_char, double pos)
{
    axis_char = (char)tolower((unsigned char)axis_char);

    if (axis_char == 't')
        return value_in_range(pos, g->t_min, g->t_max, g->dt) ? 0 : 1;

    /* spatial: map to ax1/ax2 if present */
    Axis a;
    if (axis_char_to_grid_axis(axis_char, &a) != 0)
        return 2;

    if (a == g->ax1)
        return value_in_range(pos, g->ax1_min, g->ax1_max, g->dx1) ? 0 : 1;

    if (g->has_ax2 && a == g->ax2)
        return value_in_range(pos, g->ax2_min, g->ax2_max, g->dx2) ? 0 : 1;

    /* axis not represented in cache => it must equal the fixed coordinate used to build cache */
    if (a == AXIS_X)
        return (fabs(pos - g->fixed_x) < 1e-12) ? 0 : 3;
    if (a == AXIS_Y)
        return (fabs(pos - g->fixed_y) < 1e-12) ? 0 : 3;
    if (a == AXIS_Z)
        return (fabs(pos - g->fixed_z) < 1e-12) ? 0 : 3;

    return 2;
}

/* -------------------------- slicing core --------------------------------- */

static int slice_component_from_cache(const InputSimSpec *sim,
                                      const FieldDiagSpec *d,
                                      const char *comp,
                                      const char *request_axes,
                                      float **out_data,
                                      size_t *out_n1,
                                      size_t *out_n2, /* 0 for 1D */
                                      DiagAxis *out_axis1,
                                      DiagAxis *out_axis2_or_null,
                                      DiagFixedCoords *out_fixed,
                                      char *out_units, size_t out_units_sz,
                                      char *out_label, size_t out_label_sz)
{
    if (!sim || !d || !comp || !request_axes || !out_data || !out_n1 || !out_axis1 || !out_fixed)
        return 1;

    *out_data = NULL;
    *out_n1 = 0;
    if (out_n2)
        *out_n2 = 0;
    if (out_units && out_units_sz)
        out_units[0] = '\0';
    if (out_label && out_label_sz)
        out_label[0] = '\0';

    const InputGridSpec *g = &sim->grid;
    const Axis sim_ax1 = g->ax1;
    const Axis sim_ax2 = g->ax2;
    const int has_ax2 = g->has_ax2 ? 1 : 0;

    const size_t alen = strlen(request_axes);
    if (!(alen == 1 || alen == 2))
        return 2;

    const char a0 = (char)tolower((unsigned char)request_axes[0]);
    const char a1 = (alen == 2) ? (char)tolower((unsigned char)request_axes[1]) : '\0';
    if (alen == 2 && a1 == a0)
        return 3;

    /* Validate axes are allowed by cache geometry. */
    for (size_t i = 0; i < alen; ++i)
    {
        const char ac = (char)tolower((unsigned char)request_axes[i]);
        if (ac == 't')
            continue;

        Axis asp;
        if (axis_char_to_grid_axis(ac, &asp) != 0)
            return 4;
        if (asp == sim_ax1)
            continue;
        if (has_ax2 && asp == sim_ax2)
            continue;
        return 5;
    }

    /* Open cache file: <out_dir>/<Comp>.h5 */
    char fname[64];
    snprintf(fname, sizeof(fname), "%s.h5", comp);

    char fpath[1024];
    join_path(fpath, sizeof(fpath), sim->field_cache.out_dir, fname);

    hid_t f = H5Fopen(fpath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0)
        return 10;

    hid_t dset = H5Dopen2(f, comp, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Fclose(f);
        return 11;
    }

    /* Robust: do not read UNITS/LABEL from cache; just use defaults. */
    default_units_label_for_comp(comp, out_units, out_units_sz, out_label, out_label_sz);

    hid_t fspace = H5Dget_space(dset);
    if (fspace < 0)
    {
        H5Dclose(dset);
        H5Fclose(f);
        return 12;
    }

    const int rank = H5Sget_simple_extent_ndims(fspace);
    if (!(rank == 2 || rank == 3))
    {
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 13;
    }

    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_dims(fspace, dims, NULL) < 0)
    {
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 14;
    }

    char dim_sem[3] = {'?', '?', '?'};
    const size_t t_n = (size_t)g->t_n;
    const size_t ax1_n = (size_t)g->ax1_n;
    const size_t ax2_n = (size_t)g->ax2_n;

    int irc = infer_dim_semantics(rank, dims, t_n, ax1_n, ax2_n, has_ax2, dim_sem);
    if (irc != 0)
    {
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 15;
    }

    int k_t = -1, k_1 = -1, k_2 = -1;
    if (sem_to_k('t', dim_sem, rank, &k_t) != 0)
        return 16;
    if (sem_to_k('1', dim_sem, rank, &k_1) != 0)
        return 17;
    if (rank == 3 && sem_to_k('2', dim_sem, rank, &k_2) != 0)
        return 18;

    /* Fixed indices for non-varying semantic dims. */
    const size_t idx_t = nearest_index_uniform(d->pos_t, g->t_min, g->dt, t_n);

    const char ax1c = grid_axis_to_char(sim_ax1);
    const char ax2c = grid_axis_to_char(sim_ax2);

    const size_t idx_1 = nearest_index_uniform(get_fixed_pos_for_axis_char(d, ax1c),
                                               g->ax1_min, g->dx1, ax1_n);

    const size_t idx_2 = has_ax2
                             ? nearest_index_uniform(get_fixed_pos_for_axis_char(d, ax2c), g->ax2_min, g->dx2, ax2_n)
                             : 0;

    int vary_t = 0, vary_1 = 0, vary_2 = 0;
    for (size_t i = 0; i < alen; ++i)
    {
        const char ac = (char)tolower((unsigned char)request_axes[i]);
        if (ac == 't')
        {
            vary_t = 1;
            continue;
        }

        Axis asp = AXIS_INVALID;
        if (axis_char_to_grid_axis(ac, &asp) != 0 || asp == AXIS_INVALID)
        {
            fprintf(stderr, "field_diag: invalid axis character '%c' in axes string\n", ac);
            return 1; /* or goto cleanup / continue depending on your function contract */
        }
        if (asp == sim_ax1)
            vary_1 = 1;
        else if (has_ax2 && asp == sim_ax2)
            vary_2 = 1;
    }

    if (!vary_t)
    {
        if (check_fixed_axis_in_cache(g, 't', d->pos_t) != 0)
        {
            H5Sclose(fspace);
            H5Dclose(dset);
            H5Fclose(f);
            return 40; /* out of cached t-range */
        }
    }

    if (!vary_1)
    {
        const double p1 = get_fixed_pos_for_axis_char(d, ax1c);
        if (check_fixed_axis_in_cache(g, ax1c, p1) != 0)
        {
            H5Sclose(fspace);
            H5Dclose(dset);
            H5Fclose(f);
            return 41; /* out of cached ax1-range */
        }
    }

    if (has_ax2 && !vary_2)
    {
        const double p2 = get_fixed_pos_for_axis_char(d, ax2c);
        if (check_fixed_axis_in_cache(g, ax2c, p2) != 0)
        {
            H5Sclose(fspace);
            H5Dclose(dset);
            H5Fclose(f);
            return 42; /* out of cached ax2-range */
        }
    }

    /* Output axis sizes, order preserved. */
    size_t nA0 = 0, nA1 = 0;
    if (a0 == 't')
        nA0 = t_n;
    else
    {
        Axis asp;
        (void)axis_char_to_grid_axis(a0, &asp);
        nA0 = (asp == sim_ax1) ? ax1_n : ax2_n;
    }

    if (alen == 2)
    {
        if (a1 == 't')
            nA1 = t_n;
        else
        {
            Axis asp;
            (void)axis_char_to_grid_axis(a1, &asp);
            nA1 = (asp == sim_ax1) ? ax1_n : ax2_n;
        }
    }

    /* Hyperslab selection in file space. */
    hsize_t start[3] = {0, 0, 0};
    hsize_t count[3] = {1, 1, 1};

    for (int k = 0; k < rank; ++k)
    {
        const char sem = dim_sem[k];
        if (sem == 't')
        {
            start[k] = (hsize_t)(vary_t ? 0 : idx_t);
            count[k] = (hsize_t)(vary_t ? t_n : 1);
        }
        else if (sem == '1')
        {
            start[k] = (hsize_t)(vary_1 ? 0 : idx_1);
            count[k] = (hsize_t)(vary_1 ? ax1_n : 1);
        }
        else if (sem == '2')
        {
            start[k] = (hsize_t)(vary_2 ? 0 : idx_2);
            count[k] = (hsize_t)(vary_2 ? ax2_n : 1);
        }
    }

    if (H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
    {
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 20;
    }

    hid_t mspace = H5Screate_simple(rank, count, NULL);
    if (mspace < 0)
    {
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 21;
    }

    size_t nread = 1;
    for (int k = 0; k < rank; ++k)
        nread *= (size_t)count[k];

    float *buf = (float *)malloc(nread * sizeof(float));
    if (!buf)
    {
        H5Sclose(mspace);
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 22;
    }

    if (H5Dread(dset, H5T_NATIVE_FLOAT, mspace, fspace, H5P_DEFAULT, buf) < 0)
    {
        free(buf);
        H5Sclose(mspace);
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 23;
    }

    /* Output array: axis0 fastest, axis1 slowest if 2D. */
    const size_t on1 = nA0;
    const size_t on2 = (alen == 2) ? nA1 : 0;

    const size_t out_size = on1 * (on2 ? on2 : 1);
    float *out = (float *)malloc(out_size * sizeof(float));
    if (!out)
    {
        free(buf);
        H5Sclose(mspace);
        H5Sclose(fspace);
        H5Dclose(dset);
        H5Fclose(f);
        return 24;
    }

    /* Strides for buf given count[] and file dim order. */
    size_t stride[3] = {0, 0, 0};
    stride[rank - 1] = 1;
    for (int k = rank - 2; k >= 0; --k)
        stride[k] = stride[k + 1] * (size_t)count[k + 1];

    /* Loop over output and map to buf index. */
    const size_t j2_max = on2 ? on2 : 1;

    for (size_t j2 = 0; j2 < j2_max; ++j2)
    {
        for (size_t j1 = 0; j1 < on1; ++j1)
        {
            size_t it = 0, i1s = 0, i2s = 0;

            /* Set varying axes indices from output indices. */
            set_sem_indices_for_axis(a0, j1, sim_ax1, sim_ax2, has_ax2, &it, &i1s, &i2s);
            if (alen == 2)
                set_sem_indices_for_axis(a1, j2, sim_ax1, sim_ax2, has_ax2, &it, &i1s, &i2s);

            /* Non-varying dims are 0 in the slab. */
            if (!vary_t)
                it = 0;
            if (!vary_1)
                i1s = 0;
            if (rank == 3 && !vary_2)
                i2s = 0;

            size_t bidx = 0;
            if (rank == 2)
            {
                /* sem t and 1 */
                bidx = it * stride[k_t] + i1s * stride[k_1];
            }
            else
            {
                bidx = it * stride[k_t] + i1s * stride[k_1] + i2s * stride[k_2];
            }

            out[j2 * on1 + j1] = buf[bidx];
        }
    }

    free(buf);
    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);
    H5Fclose(f);

    /* Axis metadata (DiagAxis expects min/max only here). */
    DiagAxisID did;
    if (axis_char_to_diag_id(a0, &did) != 0)
    {
        free(out);
        return 30;
    }
    out_axis1->id = did;
    out_axis1->long_name = axis_long_name_for_char(a0);
    out_axis1->units = axis_units_for_char(a0);

    if (a0 == 't')
    {
        out_axis1->vmin = g->t_min;
        out_axis1->vmax = g->t_max;
    }
    else
    {
        Axis asp;
        (void)axis_char_to_grid_axis(a0, &asp);
        if (asp == sim_ax1)
        {
            out_axis1->vmin = g->ax1_min;
            out_axis1->vmax = g->ax1_max;
        }
        else
        {
            out_axis1->vmin = g->ax2_min;
            out_axis1->vmax = g->ax2_max;
        }
    }

    if (alen == 2 && out_axis2_or_null)
    {
        if (axis_char_to_diag_id(a1, &did) != 0)
        {
            free(out);
            return 31;
        }
        out_axis2_or_null->id = did;
        out_axis2_or_null->long_name = axis_long_name_for_char(a1);
        out_axis2_or_null->units = axis_units_for_char(a1);

        if (a1 == 't')
        {
            out_axis2_or_null->vmin = g->t_min;
            out_axis2_or_null->vmax = g->t_max;
        }
        else
        {
            Axis asp;
            (void)axis_char_to_grid_axis(a1, &asp);
            if (asp == sim_ax1)
            {
                out_axis2_or_null->vmin = g->ax1_min;
                out_axis2_or_null->vmax = g->ax1_max;
            }
            else
            {
                out_axis2_or_null->vmin = g->ax2_min;
                out_axis2_or_null->vmax = g->ax2_max;
            }
        }
    }

    fill_fixed_coords_from_spec(out_fixed, d);

    *out_data = out;
    *out_n1 = on1;
    if (out_n2)
        *out_n2 = on2;

    return 0;
}

/* -------------------------- public runner --------------------------------- */

int field_diag_run_from_cache(const InputSimSpec *sim,
                              const char *out_prefix,
                              const FieldDiagOptions *opt_in,
                              MPI_Comm comm)
{
    if (!sim || !out_prefix)
        return 1;

    FieldDiagOptions opt = opt_in ? *opt_in : field_diag_default_options();

    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    const FieldDiagList *L = &sim->field_diag;
    if (!L || L->n <= 0)
    {
        MPI_Barrier(comm);
        return 0;
    }

    int first_error = 0;
    int job_idx = 0;

    for (int i = 0; i < L->n; ++i)
    {
        const FieldDiagSpec *d = &L->v[i];
        const size_t alen = strlen(d->axes);
        if (!(alen == 1 || alen == 2))
        {
            if (!first_error)
                first_error = 100;
            continue;
        }

        for (int c = 0; c < d->ncomp; ++c)
        {
            /* Independent (request, component) jobs: each reads its own cache
             * slice and writes its own distinct output file, so they can be
             * round-robined across ranks with no gather needed afterward. */
            if ((job_idx++) % nranks != rank)
                continue;

            const char *comp = d->comp[c];

            float *slice = NULL;
            size_t n1 = 0, n2 = 0;
            DiagAxis ax1, ax2;
            DiagFixedCoords fixed;
            char units[256];
            units[0] = '\0';
            char label[256];
            label[0] = '\0';

            int rc = slice_component_from_cache(sim, d, comp, d->axes,
                                                &slice, &n1, &n2,
                                                &ax1, (alen == 2 ? &ax2 : NULL),
                                                &fixed,
                                                units, sizeof(units),
                                                label, sizeof(label));
            if (rc != 0)
            {
                if (!first_error)
                    first_error = 200 + rc;

                fprintf(stderr,
                        "field_diag: request %d comp=%s axes=\"%s\" has fixed coords out of cache; "
                        "pos_t=%g pos_x=%g pos_y=%g pos_z=%g\n",
                        i, comp, d->axes, d->pos_t, d->pos_x, d->pos_y, d->pos_z);

                free(slice);
                continue;
            }

            char out_path[1024];
            build_field_diag_path(out_path, sizeof(out_path),
                                  "MS/field", comp, d->axes,
                                  &sim->grid, d);

            const double time_attr = opt.write_time_iter_0 ? 0.0 : d->pos_t;
            const int iter_attr = opt.write_time_iter_0 ? 0 : 0;

            int wrc = 0;
            if (alen == 1)
            {
                wrc = diag_h5_write_grid_1d(out_path,
                                            comp,
                                            units[0] ? units : "",
                                            label[0] ? label : comp,
                                            time_attr,
                                            iter_attr,
                                            slice,
                                            n1,
                                            &ax1,
                                            &fixed);
            }
            else
            {
                wrc = diag_h5_write_grid_2d(out_path,
                                            comp,
                                            units[0] ? units : "",
                                            label[0] ? label : comp,
                                            time_attr,
                                            iter_attr,
                                            slice,
                                            n1,
                                            n2,
                                            &ax1,
                                            &ax2,
                                            &fixed);
            }

            if (wrc != 0 && !first_error)
                first_error = 300 + wrc;

            free(slice);
        }
    }

    int global_error = 0;
    MPI_Allreduce(&first_error, &global_error, 1, MPI_INT, MPI_MAX, comm);
    return global_error;
}
