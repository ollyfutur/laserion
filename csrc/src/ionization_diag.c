#include "ionization_diag.h"

#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "diag_h5.h"
#include "ionization.h"       /* ION_compute_timeseries */
#include "ionization_model.h" /* ADKModel_init, ADK_ionization_energy */
#include "inputdeck.h"

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

/* --------------------- infer cache dimension semantics -------------------- */
/*
 * Cache dataset is rank 2 or 3 with sizes matching {t_n, ax1_n, ax2_n}.
 * We label dims as 't','1','2' corresponding to (time, sim->grid.ax1, sim->grid.ax2).
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

/*
 * Read a tile: full time, and spatial ranges:
 *   ax1: [i1_0, i1_0 + n1)
 *   ax2: [i2_0, i2_0 + n2)  (ignored if rank==2)
 *
 * Output layout: out[(j2*n1 + j1)*t_n + it]  (time fastest)
 */
static int cachecomp_read_tile(const CacheComp *cc,
                               const InputGridSpec *g,
                               size_t i1_0, size_t n1,
                               size_t i2_0, size_t n2,
                               float *out /* size n1*n2*t_n */)
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

    /* read into a raw buffer with the file's dimension order */
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

    /* strides in buf */
    size_t stride[3] = {0, 0, 0};
    stride[rank - 1] = 1;
    for (int k = rank - 2; k >= 0; --k)
        stride[k] = stride[k + 1] * (size_t)count[k + 1];

    /* map buf -> out layout (time fastest) */
    for (size_t j2 = 0; j2 < n2; ++j2)
    {
        for (size_t j1 = 0; j1 < n1; ++j1)
        {
            for (size_t it = 0; it < t_n; ++it)
            {
                size_t bidx = 0;
                if (rank == 2)
                {
                    /* semantics: t and 1 */
                    bidx = it * stride[cc->k_t] + j1 * stride[cc->k_1];
                }
                else
                {
                    bidx = it * stride[cc->k_t] + j1 * stride[cc->k_1] + j2 * stride[cc->k_2];
                }

                out[(j2 * n1 + j1) * t_n + it] = buf[bidx];
            }
        }
    }

    free(buf);
    H5Sclose(mspace);
    return 0;
}

/* -------------------------- main full-grid runner ------------------------- */

int iongrid_run_full_from_cache(const InputSimSpec *sim,
                                const char *cache_dir,
                                const char *out_dir,
                                MPI_Comm comm)
{
    if (!sim || !cache_dir || !out_dir)
        return 1;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    /* Root-only implementation for now (consistent with field_diag). */
    if (rank != 0)
    {
        MPI_Barrier(comm);
        return 0;
    }

    const InputGridSpec *g = &sim->grid;
    const size_t t_n = (size_t)g->t_n;
    const size_t ax1_n = (size_t)g->ax1_n;
    const size_t ax2_n = g->has_ax2 ? (size_t)g->ax2_n : 1u;

    /* Determine Z list from ADK tables for the chosen gas. */
    int Zmax = 0;
    if (infer_Zmax_from_tables(sim->run.gas, &Zmax) != 0)
    {
        fprintf(stderr, "ionization_diag: unsupported gas \"%s\" (no ADK table)\n", sim->run.gas);
        MPI_Barrier(comm);
        return 2;
    }

    const size_t nZ = (size_t)Zmax;
    int *Z_list = (int *)malloc(nZ * sizeof(int));
    if (!Z_list)
    {
        MPI_Barrier(comm);
        return 3;
    }
    for (size_t i = 0; i < nZ; ++i)
        Z_list[i] = (int)(i + 1);

    /* Build the ionization model (only ADK currently). */
    ADKModel adk;
    ADKModel_init(&adk);
    const IonizationModel *model = &adk.base;

    /* Precompute time array (fs). */
    double *t_fs = (double *)malloc(t_n * sizeof(double));
    if (!t_fs)
    {
        free(Z_list);
        MPI_Barrier(comm);
        return 4;
    }
    for (size_t it = 0; it < t_n; ++it)
        t_fs[it] = g->t_min + g->dt * (double)it;

    /* Output arrays are small (ax1_n * ax2_n). */
    const size_t nxy = ax1_n * ax2_n;

    float **outP = (float **)calloc(nZ, sizeof(float *));
    if (!outP)
    {
        free(t_fs);
        free(Z_list);
        MPI_Barrier(comm);
        return 5;
    }
    for (size_t iz = 0; iz < nZ; ++iz)
    {
        outP[iz] = (float *)malloc(nxy * sizeof(float));
        if (!outP[iz])
        {
            for (size_t k = 0; k < iz; ++k)
                free(outP[k]);
            free(outP);
            free(t_fs);
            free(Z_list);
            MPI_Barrier(comm);
            return 6;
        }
        for (size_t j = 0; j < nxy; ++j)
            outP[iz][j] = NAN;
    }

    float *outTot = (float *)malloc(nxy * sizeof(float));
    if (!outTot)
    {
        for (size_t iz = 0; iz < nZ; ++iz)
            free(outP[iz]);
        free(outP);
        free(t_fs);
        free(Z_list);
        MPI_Barrier(comm);
        return 7;
    }
    for (size_t j = 0; j < nxy; ++j)
        outTot[j] = NAN;

    /* Open cache components once. */
    CacheComp cEx, cEy, cEz;
    int rc = 0;

    rc = cachecomp_open(&cEx, cache_dir, "Ex", g);
    if (rc != 0)
        goto fail_open;
    rc = cachecomp_open(&cEy, cache_dir, "Ey", g);
    if (rc != 0)
        goto fail_open;
    rc = cachecomp_open(&cEz, cache_dir, "Ez", g);
    if (rc != 0)
        goto fail_open;

    /* Tile sizes (space). Keep time full. Adjust if needed. */
    const size_t tile1 = 16;
    const size_t tile2 = g->has_ax2 ? 16 : 1;

    float *bufEx = NULL, *bufEy = NULL, *bufEz = NULL;
    bufEx = (float *)malloc(tile1 * tile2 * t_n * sizeof(float));
    bufEy = (float *)malloc(tile1 * tile2 * t_n * sizeof(float));
    bufEz = (float *)malloc(tile1 * tile2 * t_n * sizeof(float));

    /* Ionization scratch (per spatial point). */
    double *Eabs = NULL;
    double *w = NULL;
    double *S = NULL;
    double *dP = NULL;
    double *Plev = NULL;

    /* allocate */
    Eabs = (double *)malloc(t_n * sizeof(double));
    w = (double *)malloc((size_t)nZ * (size_t)t_n * sizeof(double));
    S = (double *)malloc((size_t)nZ * (size_t)t_n * sizeof(double));
    dP = (double *)malloc((size_t)nZ * (size_t)t_n * sizeof(double));
    Plev = (double *)malloc((size_t)nZ * sizeof(double));

    if (!bufEx || !bufEy || !bufEz)
    {
        rc = 20;
        goto fail;
    }

    if (!Eabs || !w || !S || !dP || !Plev)
    {
        rc = 21; /* pick your error code */
        goto fail;
    }

    printf("run: ionization_frac — computing full grid from cache (%s), gas=%s, Zmax=%d\n\n",
           cache_dir, sim->run.gas, Zmax);
    fflush(stdout);

    for (size_t i2_0 = 0; i2_0 < ax2_n; i2_0 += tile2)
    {
        const size_t n2 = (i2_0 + tile2 <= ax2_n) ? tile2 : (ax2_n - i2_0);

        for (size_t i1_0 = 0; i1_0 < ax1_n; i1_0 += tile1)
        {
            const size_t n1 = (i1_0 + tile1 <= ax1_n) ? tile1 : (ax1_n - i1_0);

            /* Read tile for Ex/Ey/Ez */
            rc = cachecomp_read_tile(&cEx, g, i1_0, n1, i2_0, n2, bufEx);
            if (rc != 0)
                goto fail;
            rc = cachecomp_read_tile(&cEy, g, i1_0, n1, i2_0, n2, bufEy);
            if (rc != 0)
                goto fail;
            rc = cachecomp_read_tile(&cEz, g, i1_0, n1, i2_0, n2, bufEz);
            if (rc != 0)
                goto fail;

            /* For each spatial cell in tile, compute ionization from its time trace */
            for (size_t j2 = 0; j2 < n2; ++j2)
            {
                for (size_t j1 = 0; j1 < n1; ++j1)
                {
                    const size_t local = (j2 * n1 + j1);
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

                    double Ptot = 0.0;
                    int irc2 = ION_compute_timeseries(model,
                                                      sim->run.gas,
                                                      Z_list, nZ,
                                                      t_fs, t_n,
                                                      Eabs,
                                                      w, S, dP,
                                                      Plev,
                                                      &Ptot);
                    if (irc2 != 0)
                    {
                        /* keep NaNs for this cell */
                        continue;
                    }

                    const size_t gi1 = i1_0 + j1;
                    const size_t gi2 = i2_0 + j2;
                    const size_t gidx = gi2 * ax1_n + gi1;

                    for (size_t iz = 0; iz < nZ; ++iz)
                        outP[iz][gidx] = (float)Plev[iz];
                    outTot[gidx] = (float)Ptot;
                }
            }
        }
    }

    /* Write outputs: one file per Z and total. */
    DiagAxis ax1, ax2;
    DiagFixedCoords fixed;

    /* fixed coords are the grid-fixed for the axis not represented (if any) */
    fixed.t = 0.0;
    fixed.x = g->fixed_x;
    fixed.y = g->fixed_y;
    fixed.z = g->fixed_z;

    /* Axis 1 is sim->grid.ax1 */
    {
        char a1c = grid_axis_to_char(g->ax1);
        ax1.id = (a1c == 'x') ? DIAG_AXIS_X : (a1c == 'y') ? DIAG_AXIS_Y
                                                           : DIAG_AXIS_Z;
        ax1.long_name = (a1c == 'x') ? "x" : (a1c == 'y') ? "y"
                                                          : "z";
        ax1.units = "\\mu m";
        ax1.vmin = g->ax1_min;
        ax1.vmax = g->ax1_max;
    }

    int werr = 0;

    if (g->has_ax2)
    {
        char a2c = grid_axis_to_char(g->ax2);
        ax2.id = (a2c == 'x') ? DIAG_AXIS_X : (a2c == 'y') ? DIAG_AXIS_Y
                                                           : DIAG_AXIS_Z;
        ax2.long_name = (a2c == 'x') ? "x" : (a2c == 'y') ? "y"
                                                          : "z";
        ax2.units = "\\mu m";
        ax2.vmin = g->ax2_min;
        ax2.vmax = g->ax2_max;

        for (size_t iz = 0; iz < nZ; ++iz)
        {
            char path[1024];
            char name[64];
            snprintf(name, sizeof(name), "ion_frac_Z%02d", (int)(iz + 1));
            snprintf(path, sizeof(path), "%s/%s.h5", out_dir, name);

            int wr = diag_h5_write_grid_2d(path,
                                           name,
                                           "1",
                                           name,
                                           0.0,
                                           (int)(iz + 1),
                                           outP[iz],
                                           ax1_n,
                                           ax2_n,
                                           &ax1,
                                           &ax2,
                                           &fixed);
            if (wr != 0 && werr == 0)
                werr = 100 + wr;
        }

        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/ion_frac_Z00.h5", out_dir);
            int wr = diag_h5_write_grid_2d(path,
                                           "ion frac total",
                                           "1",
                                           "ion frac total",
                                           0.0,
                                           0,
                                           outTot,
                                           ax1_n,
                                           ax2_n,
                                           &ax1,
                                           &ax2,
                                           &fixed);
            if (wr != 0 && werr == 0)
                werr = 200 + wr;
        }
    }
    else
    {
        for (size_t iz = 0; iz < nZ; ++iz)
        {
            char path[1024];
            char name[64];
            snprintf(name, sizeof(name), "ion_frac_Z%02d", (int)(iz + 1));
            snprintf(path, sizeof(path), "%s/%s.h5", out_dir, name);

            int wr = diag_h5_write_grid_1d(path,
                                           name,
                                           "1",
                                           name,
                                           0.0,
                                           (int)(iz + 1),
                                           outP[iz],
                                           ax1_n,
                                           &ax1,
                                           &fixed);
            if (wr != 0 && werr == 0)
                werr = 100 + wr;
        }

        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/ion_frac_Z00.h5", out_dir);
            int wr = diag_h5_write_grid_1d(path,
                                           "ion frac total",
                                           "1",
                                           "ion frac total",
                                           0.0,
                                           0,
                                           outTot,
                                           ax1_n,
                                           &ax1,
                                           &fixed);
            if (wr != 0 && werr == 0)
                werr = 200 + wr;
        }
    }

    rc = werr; /* 0 if OK */

fail:
    free(Plev);
    free(dP);
    free(S);
    free(w);
    free(Eabs);
    free(bufEz);
    free(bufEy);
    free(bufEx);
    cachecomp_close(&cEz);
    cachecomp_close(&cEy);
    cachecomp_close(&cEx);

fail_open:
    for (size_t iz = 0; iz < nZ; ++iz)
        free(outP[iz]);
    free(outP);
    free(outTot);
    free(t_fs);
    free(Z_list);

    MPI_Barrier(comm);
    return rc;
}
