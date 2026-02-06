#include "mdf_diag.h"

#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "diag_h5.h"
#include "ionization.h"
#include "ionization_model.h"
#include "mdf.h" /* MDF_CONV_A_TO_P */

#define MDF_DIAG_SKIPPED_NO_A 777001

/* ------------------------------ small utils ------------------------------ */

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

static const char *comp_to_str(int c) /* 0:x,1:y,2:z */
{
    return (c == 0) ? "x" : (c == 1) ? "y"
                                     : "z";
}

static DiagAxisID comp_to_diag_axis_id(int c)
{
    return (c == 0) ? DIAG_AXIS_X : (c == 1) ? DIAG_AXIS_Y
                                             : DIAG_AXIS_Z;
}

/* --------------------- infer cache dimension semantics -------------------- */
/* Cache dataset is rank 2 or 3 with sizes matching {t_n, ax1_n, ax2_n}. */
static int infer_dim_semantics(int rank,
                               const hsize_t *dims,
                               size_t t_n,
                               size_t ax1_n,
                               size_t ax2_n,
                               int has_ax2,
                               char *dim_sem)
{
    if (!dims || !dim_sem)
        return 1;
    if (!(rank == 2 || rank == 3))
        return 2;

    for (int k = 0; k < rank; ++k)
        dim_sem[k] = '?';

    int used_t = 0, used_1 = 0, used_2 = 0;

    for (int k = 0; k < rank; ++k)
        if (!used_t && (size_t)dims[k] == t_n)
        {
            dim_sem[k] = 't';
            used_t = 1;
        }

    for (int k = 0; k < rank; ++k)
        if (dim_sem[k] == '?' && !used_1 && (size_t)dims[k] == ax1_n)
        {
            dim_sem[k] = '1';
            used_1 = 1;
        }

    for (int k = 0; k < rank; ++k)
        if (dim_sem[k] == '?' && rank == 3 && has_ax2 && !used_2 && (size_t)dims[k] == ax2_n)
        {
            dim_sem[k] = '2';
            used_2 = 1;
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
        if (dim_sem[k] == sem)
        {
            if (out_k)
                *out_k = k;
            return 0;
        }
    return 1;
}

/* ------------------------- Z-list inference (ADK tables) ------------------ */

static int infer_Zmax_from_tables(const char *species, int *Zmax_out)
{
    if (!species || !Zmax_out)
        return 1;

    int Z = 1;
    for (;;)
    {
        double E = 0.0;
        int rc = ADK_ionization_energy(&E, species, Z);
        if (rc != 0)
            break;
        ++Z;
        if (Z > 256)
            return 2;
    }
    *Zmax_out = Z - 1;
    return (*Zmax_out > 0) ? 0 : 3;
}

/* ---------------------------- HDF5 cache reader --------------------------- */

typedef struct CacheComp
{
    hid_t f;
    hid_t dset;
    hid_t fspace;
    int rank;
    hsize_t dims[3];
    char dim_sem[3];
    int k_t, k_1, k_2;
} CacheComp;

static void cachecomp_close(CacheComp *c)
{
    if (!c)
        return;
    if (c->fspace >= 0)
        H5Sclose(c->fspace);
    if (c->dset >= 0)
        H5Dclose(c->dset);
    if (c->f >= 0)
        H5Fclose(c->f);
    c->f = c->dset = c->fspace = -1;
}

static int cachecomp_open(CacheComp *cc,
                          const char *cache_dir,
                          const char *comp, /* "Ex" */
                          const InputGridSpec *g)
{
    memset(cc, 0, sizeof(*cc));
    cc->f = cc->dset = cc->fspace = -1;
    cc->rank = 0;
    cc->dims[0] = cc->dims[1] = cc->dims[2] = 0;
    cc->dim_sem[0] = cc->dim_sem[1] = cc->dim_sem[2] = '?';
    cc->k_t = cc->k_1 = cc->k_2 = -1;

    char fname[64];
    snprintf(fname, sizeof(fname), "%s.h5", comp);

    char fpath[1024];
    join_path(fpath, sizeof(fpath), cache_dir, fname);

    cc->f = H5Fopen(fpath, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (cc->f < 0)
        return 10;

    cc->dset = H5Dopen2(cc->f, comp, H5P_DEFAULT);
    if (cc->dset < 0)
    {
        cachecomp_close(cc);
        return 11;
    }

    cc->fspace = H5Dget_space(cc->dset);
    if (cc->fspace < 0)
    {
        cachecomp_close(cc);
        return 12;
    }

    cc->rank = H5Sget_simple_extent_ndims(cc->fspace);
    if (!(cc->rank == 2 || cc->rank == 3))
    {
        cachecomp_close(cc);
        return 13;
    }

    if (H5Sget_simple_extent_dims(cc->fspace, cc->dims, NULL) < 0)
    {
        cachecomp_close(cc);
        return 14;
    }

    const size_t t_n = (size_t)g->t_n;
    const size_t ax1_n = (size_t)g->ax1_n;
    const size_t ax2_n = (size_t)g->ax2_n;

    int irc = infer_dim_semantics(cc->rank, cc->dims, t_n, ax1_n, ax2_n, g->has_ax2 ? 1 : 0, cc->dim_sem);
    if (irc != 0)
    {
        cachecomp_close(cc);
        return 15;
    }

    if (sem_to_k('t', cc->dim_sem, cc->rank, &cc->k_t) != 0)
    {
        cachecomp_close(cc);
        return 16;
    }
    if (sem_to_k('1', cc->dim_sem, cc->rank, &cc->k_1) != 0)
    {
        cachecomp_close(cc);
        return 17;
    }
    if (cc->rank == 3)
        if (sem_to_k('2', cc->dim_sem, cc->rank, &cc->k_2) != 0)
        {
            cachecomp_close(cc);
            return 18;
        }

    return 0;
}

/* Read a tile (time full), output layout: out[(j2*n1+j1)*t_n + it] (time fastest). */
static int cachecomp_read_tile(const CacheComp *cc,
                               const InputGridSpec *g,
                               size_t i1_0, size_t n1,
                               size_t i2_0, size_t n2,
                               float *out)
{
    const size_t t_n = (size_t)g->t_n;
    const int rank = cc->rank;

    hsize_t start[3] = {0, 0, 0};
    hsize_t count[3] = {1, 1, 1};

    for (int k = 0; k < rank; ++k)
    {
        const char sem = cc->dim_sem[k];
        if (sem == 't')
        {
            start[k] = 0;
            count[k] = (hsize_t)t_n;
        }
        if (sem == '1')
        {
            start[k] = (hsize_t)i1_0;
            count[k] = (hsize_t)n1;
        }
        if (sem == '2')
        {
            start[k] = (hsize_t)i2_0;
            count[k] = (hsize_t)n2;
        }
    }

    if (H5Sselect_hyperslab(cc->fspace, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
        return 20;

    hid_t mspace = H5Screate_simple(rank, count, NULL);
    if (mspace < 0)
        return 21;

    size_t nread = 1;
    for (int k = 0; k < rank; ++k)
        nread *= (size_t)count[k];

    float *buf = (float *)malloc(nread * sizeof(float));
    if (!buf)
    {
        H5Sclose(mspace);
        return 22;
    }

    if (H5Dread(cc->dset, H5T_NATIVE_FLOAT, mspace, cc->fspace, H5P_DEFAULT, buf) < 0)
    {
        free(buf);
        H5Sclose(mspace);
        return 23;
    }

    size_t stride[3] = {0, 0, 0};
    stride[rank - 1] = 1;
    for (int k = rank - 2; k >= 0; --k)
        stride[k] = stride[k + 1] * (size_t)count[k + 1];

    for (size_t j2 = 0; j2 < n2; ++j2)
    {
        for (size_t j1 = 0; j1 < n1; ++j1)
        {
            for (size_t it = 0; it < t_n; ++it)
            {
                size_t bidx = 0;
                if (rank == 2)
                    bidx = it * stride[cc->k_t] + j1 * stride[cc->k_1];
                else
                    bidx = it * stride[cc->k_t] + j1 * stride[cc->k_1] + j2 * stride[cc->k_2];

                out[(j2 * n1 + j1) * t_n + it] = buf[bidx];
            }
        }
    }

    free(buf);
    H5Sclose(mspace);
    return 0;
}

/* ------------------------------ histogram utils -------------------------- */

static int clamp_bin(double v, double vmin, double vmax, int nbins)
{
    if (!(vmax > vmin) || nbins <= 0)
        return -1;
    if (v < vmin || v >= vmax)
        return -1;

    const double u = (v - vmin) / (vmax - vmin);
    int b = (int)floor(u * (double)nbins);

    if (b < 0)
        b = 0;
    if (b >= nbins)
        b = nbins - 1;
    return b;
}

static void phasespace_kind_components(PhaseSpaceKind k, int *c1, int *c2, int *is2d)
{
    *c1 = 0;
    *c2 = 0;
    *is2d = 0;

    switch (k)
    {
    case PHASESPACE_PX:
        *c1 = 0;
        *is2d = 0;
        break;
    case PHASESPACE_PY:
        *c1 = 1;
        *is2d = 0;
        break;
    case PHASESPACE_PZ:
        *c1 = 2;
        *is2d = 0;
        break;

    case PHASESPACE_PX_PY:
        *c1 = 0;
        *c2 = 1;
        *is2d = 1;
        break;
    case PHASESPACE_PY_PX:
        *c1 = 1;
        *c2 = 0;
        *is2d = 1;
        break;

    case PHASESPACE_PX_PZ:
        *c1 = 0;
        *c2 = 2;
        *is2d = 1;
        break;
    case PHASESPACE_PZ_PX:
        *c1 = 2;
        *c2 = 0;
        *is2d = 1;
        break;

    case PHASESPACE_PY_PZ:
        *c1 = 1;
        *c2 = 2;
        *is2d = 1;
        break;
    case PHASESPACE_PZ_PY:
        *c1 = 2;
        *c2 = 1;
        *is2d = 1;
        break;

    default:
        *c1 = 0;
        *is2d = 0;
        break;
    }
}

static void kind_to_strings(PhaseSpaceKind k, char *kindbuf, size_t kindbuf_sz)
{
    switch (k)
    {
    case PHASESPACE_PX:
        snprintf(kindbuf, kindbuf_sz, "px");
        break;
    case PHASESPACE_PY:
        snprintf(kindbuf, kindbuf_sz, "py");
        break;
    case PHASESPACE_PZ:
        snprintf(kindbuf, kindbuf_sz, "pz");
        break;

    case PHASESPACE_PX_PY:
        snprintf(kindbuf, kindbuf_sz, "px_py");
        break;
    case PHASESPACE_PY_PX:
        snprintf(kindbuf, kindbuf_sz, "py_px");
        break;

    case PHASESPACE_PX_PZ:
        snprintf(kindbuf, kindbuf_sz, "px_pz");
        break;
    case PHASESPACE_PZ_PX:
        snprintf(kindbuf, kindbuf_sz, "pz_px");
        break;

    case PHASESPACE_PY_PZ:
        snprintf(kindbuf, kindbuf_sz, "py_pz");
        break;
    case PHASESPACE_PZ_PY:
        snprintf(kindbuf, kindbuf_sz, "pz_py");
        break;

    default:
        snprintf(kindbuf, kindbuf_sz, "px");
        break;
    }
}

static void build_label(PhaseSpaceKind k, char *label, size_t label_sz)
{
    int c1 = 0, c2 = 0, is2d = 0;
    phasespace_kind_components(k, &c1, &c2, &is2d);

    if (!is2d)
        snprintf(label, label_sz, "f(p_%s)", comp_to_str(c1));
    else
        snprintf(label, label_sz, "f(p_%s,p_%s)", comp_to_str(c1), comp_to_str(c2));
}

/* For cache-based postprocessing: skip cell if max(|E|) < envelope_cut (if envelope_cut>0). */
static int passes_envelope_cut(const double *Eabs, size_t t_n, double envelope_cut)
{
    if (!(envelope_cut > 0.0))
        return 1;
    double mx = 0.0;
    for (size_t i = 0; i < t_n; ++i)
        if (Eabs[i] > mx)
            mx = Eabs[i];
    return (mx >= envelope_cut) ? 1 : 0;
}

/* ------------------------------ region cropping -------------------------- */
/* We assume the cache grid points are cell-centers:
 *   x_i = min + (i+0.5)*dx
 * Find i range such that x_i in [rmin, rmax].
 */
static void axis_crop_1d(double ax_min, double dx, int n,
                         double rmin, double rmax,
                         int *i_lo, int *i_hi_excl)
{
    int lo = 0, hi = n;
    if (!(rmax > rmin))
    {
        *i_lo = 0;
        *i_hi_excl = 0;
        return;
    }

    const double a = (rmin - ax_min) / dx - 0.5;
    const double b = (rmax - ax_min) / dx - 0.5;

    lo = (int)ceil(a);
    hi = (int)floor(b) + 1;

    if (lo < 0)
        lo = 0;
    if (hi > n)
        hi = n;
    if (hi < lo)
        hi = lo;

    *i_lo = lo;
    *i_hi_excl = hi;
}

static int fixed_coord_inside(double v, double vmin, double vmax)
{
    if (!(vmax > vmin))
        return 1;
    return (v >= vmin && v <= vmax) ? 1 : 0;
}

/* Determine cropped spatial index ranges on the swept axes.
 * Returns:
 *   1 if there are points to process, 0 if region excludes everything.
 */
static int compute_crop_ranges(const InputGridSpec *g,
                               const PhaseSpaceSpec *ps,
                               int *i1_lo, int *i1_hi,
                               int *i2_lo, int *i2_hi)
{
    *i1_lo = 0;
    *i1_hi = g->ax1_n;
    *i2_lo = 0;
    *i2_hi = g->has_ax2 ? g->ax2_n : 1;

    if (!ps->has_region)
        return 1;

    int swept_x = 0, swept_y = 0, swept_z = 0;

    if (g->ax1 == AXIS_X)
        swept_x = 1;
    if (g->ax1 == AXIS_Y)
        swept_y = 1;
    if (g->ax1 == AXIS_Z)
        swept_z = 1;

    if (g->has_ax2)
    {
        if (g->ax2 == AXIS_X)
            swept_x = 1;
        if (g->ax2 == AXIS_Y)
            swept_y = 1;
        if (g->ax2 == AXIS_Z)
            swept_z = 1;
    }

    if (!swept_x && !fixed_coord_inside(g->fixed_x, ps->xmin, ps->xmax))
        return 0;
    if (!swept_y && !fixed_coord_inside(g->fixed_y, ps->ymin, ps->ymax))
        return 0;
    if (!swept_z && !fixed_coord_inside(g->fixed_z, ps->zmin, ps->zmax))
        return 0;

    if (g->ax1 == AXIS_X)
        axis_crop_1d(g->ax1_min, g->dx1, g->ax1_n, ps->xmin, ps->xmax, i1_lo, i1_hi);
    if (g->ax1 == AXIS_Y)
        axis_crop_1d(g->ax1_min, g->dx1, g->ax1_n, ps->ymin, ps->ymax, i1_lo, i1_hi);
    if (g->ax1 == AXIS_Z)
        axis_crop_1d(g->ax1_min, g->dx1, g->ax1_n, ps->zmin, ps->zmax, i1_lo, i1_hi);

    if (g->has_ax2)
    {
        if (g->ax2 == AXIS_X)
            axis_crop_1d(g->ax2_min, g->dx2, g->ax2_n, ps->xmin, ps->xmax, i2_lo, i2_hi);
        if (g->ax2 == AXIS_Y)
            axis_crop_1d(g->ax2_min, g->dx2, g->ax2_n, ps->ymin, ps->ymax, i2_lo, i2_hi);
        if (g->ax2 == AXIS_Z)
            axis_crop_1d(g->ax2_min, g->dx2, g->ax2_n, ps->zmin, ps->zmax, i2_lo, i2_hi);
    }
    else
    {
        *i2_lo = 0;
        *i2_hi = 1;
    }

    if (*i1_hi <= *i1_lo)
        return 0;
    if (g->has_ax2 && *i2_hi <= *i2_lo)
        return 0;
    return 1;
}

/* -------------------------- core: one diagnostic -------------------------- */

int mdf_diag_run_one_from_cache(const InputSimSpec *sim,
                                const PhaseSpaceSpec *ps,
                                const char *cache_dir,
                                const char *out_dir,
                                MPI_Comm comm)
{
    if (!sim || !ps || !cache_dir || !out_dir)
        return 1;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int rc = 0;

    /* Make all frees safe even if we jump early. */
    int *Z_list = NULL;
    double *t_fs = NULL;
    double *hist_local = NULL;
    double *hist_global = NULL;

    float *bufEx = NULL, *bufEy = NULL, *bufEz = NULL;
    float *bufAx = NULL, *bufAy = NULL, *bufAz = NULL;

    double *Eabs = NULL;
    double *w = NULL;
    double *S = NULL;
    double *dP = NULL;
    double *Plev = NULL;

    CacheComp cEx, cEy, cEz;
    CacheComp cAx, cAy, cAz;
    memset(&cEx, 0, sizeof(cEx));
    memset(&cEy, 0, sizeof(cEy));
    memset(&cEz, 0, sizeof(cEz));
    memset(&cAx, 0, sizeof(cAx));
    memset(&cAy, 0, sizeof(cAy));
    memset(&cAz, 0, sizeof(cAz));
    cEx.f = cEx.dset = cEx.fspace = -1;
    cEy.f = cEy.dset = cEy.fspace = -1;
    cEz.f = cEz.dset = cEz.fspace = -1;
    cAx.f = cAx.dset = cAx.fspace = -1;
    cAy.f = cAy.dset = cAy.fspace = -1;
    cAz.f = cAz.dset = cAz.fspace = -1;

    const InputGridSpec *g = &sim->grid;
    const size_t t_n = (size_t)g->t_n;

    int Zmax = 0;
    if (infer_Zmax_from_tables(sim->run.gas, &Zmax) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "mdf_diag: unsupported gas \"%s\" (no ADK table)\n", sim->run.gas);
        rc = 2;
        goto fail;
    }

    const size_t nZ = (size_t)Zmax;
    Z_list = (int *)malloc(nZ * sizeof(int));
    if (!Z_list)
    {
        rc = 3;
        goto fail;
    }
    for (size_t i = 0; i < nZ; ++i)
        Z_list[i] = (int)(i + 1);

    ADKModel adk;
    ADKModel_init(&adk);
    const IonizationModel *model = &adk.base;

    t_fs = (double *)malloc(t_n * sizeof(double));
    if (!t_fs)
    {
        rc = 4;
        goto fail;
    }
    for (size_t it = 0; it < t_n; ++it)
        t_fs[it] = g->t_min + g->dt * (double)it;

    int c1 = 0, c2 = 0, is2d = 0;
    phasespace_kind_components(ps->kind, &c1, &c2, &is2d);

    const int n1 = ps->nbins1;
    const int n2 = is2d ? ps->nbins2 : 1;
    const size_t hist_sz = (size_t)n1 * (size_t)n2;

    hist_local = (double *)calloc(hist_sz, sizeof(double));
    hist_global = (double *)calloc(hist_sz, sizeof(double));
    if (!hist_local || !hist_global)
    {
        rc = 5;
        goto fail;
    }

    int i1_lo = 0, i1_hi = 0, i2_lo = 0, i2_hi = 0;
    const int has_points = compute_crop_ranges(g, ps, &i1_lo, &i1_hi, &i2_lo, &i2_hi);

    /* Open E cache (needed for envelope_cut + ionization). */
    rc = cachecomp_open(&cEx, cache_dir, "Ex", g);
    if (rc != 0)
        goto fail;
    rc = cachecomp_open(&cEy, cache_dir, "Ey", g);
    if (rc != 0)
        goto fail;
    rc = cachecomp_open(&cEz, cache_dir, "Ez", g);
    if (rc != 0)
        goto fail;

    /* Require A cache. Decide collectively to avoid MPI divergence. */
    int have_A = 1;
    if (rank == 0)
    {
        CacheComp tAx, tAy, tAz;
        memset(&tAx, 0, sizeof(tAx));
        memset(&tAy, 0, sizeof(tAy));
        memset(&tAz, 0, sizeof(tAz));
        tAx.f = tAx.dset = tAx.fspace = -1;
        tAy.f = tAy.dset = tAy.fspace = -1;
        tAz.f = tAz.dset = tAz.fspace = -1;

        have_A = 1;
        if (cachecomp_open(&tAx, cache_dir, "Ax", g) != 0)
            have_A = 0;
        if (cachecomp_open(&tAy, cache_dir, "Ay", g) != 0)
            have_A = 0;
        if (cachecomp_open(&tAz, cache_dir, "Az", g) != 0)
            have_A = 0;

        cachecomp_close(&tAz);
        cachecomp_close(&tAy);
        cachecomp_close(&tAx);
    }
    MPI_Bcast(&have_A, 1, MPI_INT, 0, comm);

    if (!have_A)
    {
        if (rank == 0)
        {
            char kindbuf[32];
            kind_to_strings(ps->kind, kindbuf, sizeof(kindbuf));
            printf("mdf_diag: skipping %s (Ax/Ay/Az not present in cache: %s)\n", kindbuf, cache_dir);
            fflush(stdout);
        }
        rc = MDF_DIAG_SKIPPED_NO_A;
        goto fail;
    }

    /* Now open A on all ranks (serial HDF5). */
    rc = cachecomp_open(&cAx, cache_dir, "Ax", g);
    if (rc != 0)
        goto fail;
    rc = cachecomp_open(&cAy, cache_dir, "Ay", g);
    if (rc != 0)
        goto fail;
    rc = cachecomp_open(&cAz, cache_dir, "Az", g);
    if (rc != 0)
        goto fail;

    if (rank == 0)
    {
        char kindbuf[32];
        kind_to_strings(ps->kind, kindbuf, sizeof(kindbuf));
        printf("run: mdf_diag — %s (%s), MPI ranks=%d, region=%s, using_A_cache\n",
               kindbuf, cache_dir, size, ps->has_region ? "ON" : "OFF");
        fflush(stdout);
    }

    if (!has_points)
    {
        MPI_Allreduce(hist_local, hist_global, (int)hist_sz, MPI_DOUBLE, MPI_SUM, comm);
        goto write_out;
    }

    const size_t tile1 = 16;
    const size_t tile2 = g->has_ax2 ? 16 : 1;
    const size_t max_tile_elems = tile1 * tile2 * t_n;

    bufEx = (float *)malloc(max_tile_elems * sizeof(float));
    bufEy = (float *)malloc(max_tile_elems * sizeof(float));
    bufEz = (float *)malloc(max_tile_elems * sizeof(float));
    bufAx = (float *)malloc(max_tile_elems * sizeof(float));
    bufAy = (float *)malloc(max_tile_elems * sizeof(float));
    bufAz = (float *)malloc(max_tile_elems * sizeof(float));
    if (!bufEx || !bufEy || !bufEz || !bufAx || !bufAy || !bufAz)
    {
        rc = 10;
        goto fail;
    }

    Eabs = (double *)malloc(t_n * sizeof(double));
    w = (double *)malloc(nZ * t_n * sizeof(double));
    S = (double *)malloc(nZ * t_n * sizeof(double));
    dP = (double *)malloc(nZ * t_n * sizeof(double));
    Plev = (double *)malloc(nZ * sizeof(double));
    if (!Eabs || !w || !S || !dP || !Plev)
    {
        rc = 11;
        goto fail;
    }

    long long tile_linear = 0;

    for (int i2_0 = i2_lo; i2_0 < i2_hi; i2_0 += (int)tile2)
    {
        const size_t n2t = (size_t)(((i2_0 + (int)tile2) <= i2_hi) ? tile2 : (size_t)(i2_hi - i2_0));

        for (int i1_0 = i1_lo; i1_0 < i1_hi; i1_0 += (int)tile1)
        {
            const size_t n1t = (size_t)(((i1_0 + (int)tile1) <= i1_hi) ? tile1 : (size_t)(i1_hi - i1_0));

            const int owner = (int)(tile_linear % (long long)size);
            ++tile_linear;
            if (owner != rank)
                continue;

            rc = cachecomp_read_tile(&cEx, g, (size_t)i1_0, n1t, (size_t)i2_0, n2t, bufEx);
            if (rc != 0)
                goto fail;
            rc = cachecomp_read_tile(&cEy, g, (size_t)i1_0, n1t, (size_t)i2_0, n2t, bufEy);
            if (rc != 0)
                goto fail;
            rc = cachecomp_read_tile(&cEz, g, (size_t)i1_0, n1t, (size_t)i2_0, n2t, bufEz);
            if (rc != 0)
                goto fail;

            rc = cachecomp_read_tile(&cAx, g, (size_t)i1_0, n1t, (size_t)i2_0, n2t, bufAx);
            if (rc != 0)
                goto fail;
            rc = cachecomp_read_tile(&cAy, g, (size_t)i1_0, n1t, (size_t)i2_0, n2t, bufAy);
            if (rc != 0)
                goto fail;
            rc = cachecomp_read_tile(&cAz, g, (size_t)i1_0, n1t, (size_t)i2_0, n2t, bufAz);
            if (rc != 0)
                goto fail;

            for (size_t j2 = 0; j2 < n2t; ++j2)
            {
                for (size_t j1 = 0; j1 < n1t; ++j1)
                {
                    const size_t local = (j2 * n1t + j1);

                    const float *ex = &bufEx[local * t_n];
                    const float *ey = &bufEy[local * t_n];
                    const float *ez = &bufEz[local * t_n];

                    for (size_t it = 0; it < t_n; ++it)
                    {
                        const double Ex = (double)ex[it];
                        const double Ey = (double)ey[it];
                        const double Ez = (double)ez[it];
                        Eabs[it] = sqrt(Ex * Ex + Ey * Ey + Ez * Ez);
                    }

                    if (!passes_envelope_cut(Eabs, t_n, ps->envelope_cut))
                        continue;

                    double Ptot = 0.0;
                    const int irc2 = ION_compute_timeseries(model,
                                                            sim->run.gas,
                                                            Z_list, nZ,
                                                            t_fs, t_n,
                                                            Eabs,
                                                            w, S, dP,
                                                            Plev,
                                                            &Ptot);
                    if (irc2 != 0)
                        continue;

                    const float *ax_f = &bufAx[local * t_n];
                    const float *ay_f = &bufAy[local * t_n];
                    const float *az_f = &bufAz[local * t_n];

                    for (size_t l = 0; l < nZ; ++l)
                    {
                        const double *dPl = &dP[l * t_n];

                        for (size_t it = 0; it < t_n; ++it)
                        {
                            const double wgt = dPl[it];
                            if (!(wgt > 0.0))
                                continue;

                            const double pvec[3] = {
                                (double)ax_f[it] * MDF_CONV_A_TO_P,
                                (double)ay_f[it] * MDF_CONV_A_TO_P,
                                (double)az_f[it] * MDF_CONV_A_TO_P};

                            const double p1 = pvec[c1];
                            const int b1 = clamp_bin(p1, ps->p1min, ps->p1max, n1);
                            if (b1 < 0)
                                continue;

                            if (!is2d)
                            {
                                hist_local[(size_t)b1] += wgt;
                            }
                            else
                            {
                                const double p2 = pvec[c2];
                                const int b2 = clamp_bin(p2, ps->p2min, ps->p2max, n2);
                                if (b2 < 0)
                                    continue;

                                hist_local[(size_t)b2 * (size_t)n1 + (size_t)b1] += wgt;
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Allreduce(hist_local, hist_global, (int)hist_sz, MPI_DOUBLE, MPI_SUM, comm);

    if (ps->normalize_sum_to_1)
    {
        double sum = 0.0;
        for (size_t i = 0; i < hist_sz; ++i)
            sum += hist_global[i];
        if (sum > 0.0)
            for (size_t i = 0; i < hist_sz; ++i)
                hist_global[i] /= sum;
    }

write_out:
    /* Root writes */
    if (rank == 0)
    {
        char kindbuf[32];
        kind_to_strings(ps->kind, kindbuf, sizeof(kindbuf));

        char dataset[64];
        snprintf(dataset, sizeof(dataset), "f_%s", kindbuf);

        char label[128];
        build_label(ps->kind, label, sizeof(label));

        char path[1024];
        snprintf(path, sizeof(path), "%s/mdf_%s.h5", out_dir, kindbuf);

        float *data_f32 = (float *)malloc(hist_sz * sizeof(float));
        if (!data_f32)
        {
            rc = 30;
        }
        else
        {
            for (size_t i = 0; i < hist_sz; ++i)
                data_f32[i] = (float)hist_global[i];

            DiagFixedCoords fixed;
            fixed.t = 0.0;
            fixed.x = g->fixed_x;
            fixed.y = g->fixed_y;
            fixed.z = g->fixed_z;

            DiagAxis ax1, ax2;
            ax1.id = comp_to_diag_axis_id(c1);
            {
                static char name1[16];
                snprintf(name1, sizeof(name1), "p_%s", comp_to_str(c1));
                ax1.long_name = name1;
            }
            ax1.units = "m_e c";
            ax1.vmin = ps->p1min;
            ax1.vmax = ps->p1max;

            if (!is2d)
            {
                rc = diag_h5_write_grid_1d(path,
                                           dataset,
                                           "1",
                                           label,
                                           0.0, 0,
                                           data_f32,
                                           (size_t)n1,
                                           &ax1,
                                           &fixed);
            }
            else
            {
                ax2.id = comp_to_diag_axis_id(c2);
                {
                    static char name2[16];
                    snprintf(name2, sizeof(name2), "p_%s", comp_to_str(c2));
                    ax2.long_name = name2;
                }
                ax2.units = "m_e c";
                ax2.vmin = ps->p2min;
                ax2.vmax = ps->p2max;

                rc = diag_h5_write_grid_2d(path,
                                           dataset,
                                           "1",
                                           label,
                                           0.0, 0,
                                           data_f32,
                                           (size_t)n1, (size_t)n2,
                                           &ax1, &ax2,
                                           &fixed);
            }

            free(data_f32);
        }
    }

    /* Broadcast rc so all ranks return consistent code */
    MPI_Bcast(&rc, 1, MPI_INT, 0, comm);

fail:
    /* Free scratch/buffers (free(NULL) is OK). */
    free(Plev);
    free(dP);
    free(S);
    free(w);
    free(Eabs);

    free(bufAz);
    free(bufAy);
    free(bufAx);
    free(bufEz);
    free(bufEy);
    free(bufEx);

    cachecomp_close(&cAz);
    cachecomp_close(&cAy);
    cachecomp_close(&cAx);

    cachecomp_close(&cEz);
    cachecomp_close(&cEy);
    cachecomp_close(&cEx);

    free(hist_global);
    free(hist_local);
    free(t_fs);
    free(Z_list);

    MPI_Barrier(comm);
    return rc;
}

int mdf_diag_run_all_from_cache(const InputSimSpec *sim,
                                const char *cache_dir,
                                const char *out_dir,
                                MPI_Comm comm)
{
    if (!sim || !cache_dir || !out_dir)
        return 1;

    int rc = 0;
    for (int i = 0; i < sim->phase_space.n; ++i)
    {
        const PhaseSpaceSpec *ps = &sim->phase_space.v[i];
        int one = mdf_diag_run_one_from_cache(sim, ps, cache_dir, out_dir, comm);
        if (one == MDF_DIAG_SKIPPED_NO_A)
            continue; /* ignore */
        if (one != 0 && rc == 0)
            rc = 100 + one;
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
        printf("\n");
    MPI_Barrier(comm);
    return rc;
}
