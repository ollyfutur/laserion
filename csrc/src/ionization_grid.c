#include "ionization_grid.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ionization.h" /* ION_probs_at_r */

typedef struct IonGridCtx
{
    const LaserPulse *pulse;
    const IonGridOptions *opt;
    double *P_levels; /* length nZ (scratch) */
} IonGridCtx;

static int iongrid_eval(const DG_FixedCoords *c, void *ctx_void,
                        float *out_vals, size_t ndatasets)
{
    IonGridCtx *ctx = (IonGridCtx *)ctx_void;
    const IonGridOptions *opt = ctx->opt;

    (void)ndatasets;

    const double r_um[3] = {c->x_um, c->y_um, c->z_um};

    double Ptot = 0.0;
    const int rc = ION_probs_at_r(ctx->pulse,
                                  opt->species,
                                  opt->Z_list, opt->nZ,
                                  r_um,
                                  opt->ion_model,
                                  opt->tmin_fs,
                                  opt->tmax_fs,
                                  opt->dt_fs,
                                  ctx->P_levels,
                                  (opt->write_total ? &Ptot : NULL));
    if (rc != 0) return rc;

    for (size_t iz = 0; iz < opt->nZ; ++iz)
        out_vals[iz] = (float)ctx->P_levels[iz];

    if (opt->write_total)
        out_vals[opt->nZ] = (float)Ptot;

    return 0;
}

static void make_suffix(char *buf, size_t bufsz, const char *kind, int Z)
{
    if (strcmp(kind, "Z") == 0)
        snprintf(buf, bufsz, "_Pion_Z%d.h5", Z);
    else
        snprintf(buf, bufsz, "_Pion_total.h5");
}

int iongrid_run_1d(const LaserPulse *pulse,
                   const DG_Request1D *req,
                   const IonGridOptions *ion,
                   const char *path_prefix,
                   MPI_Comm comm)
{
    if (!pulse || !req || !ion || !path_prefix) return 1;
    if (!ion->species || !ion->Z_list || ion->nZ == 0) return 2;
    if (!(ion->dt_fs > 0.0) || !(ion->tmax_fs > ion->tmin_fs)) return 3;

    const size_t ndatasets = ion->nZ + (ion->write_total ? 1u : 0u);

    DG_DatasetDesc *ds = (DG_DatasetDesc *)calloc(ndatasets, sizeof(DG_DatasetDesc));
    if (!ds) return 4;

    char **suffix_store = (char **)calloc(ndatasets, sizeof(char *));
    char **name_store   = (char **)calloc(ndatasets, sizeof(char *));
    if (!suffix_store || !name_store)
    {
        free(ds); free(suffix_store); free(name_store);
        return 5;
    }

    for (size_t i = 0; i < ndatasets; ++i)
    {
        suffix_store[i] = (char *)calloc(64, 1);
        name_store[i]   = (char *)calloc(64, 1);
        if (!suffix_store[i] || !name_store[i])
        {
            for (size_t k = 0; k <= i; ++k) { free(suffix_store[k]); free(name_store[k]); }
            free(ds); free(suffix_store); free(name_store);
            return 6;
        }
    }

    for (size_t iz = 0; iz < ion->nZ; ++iz)
    {
        const int Z = ion->Z_list[iz];
        snprintf(name_store[iz], 64, "Pion_Z%d", Z);
        make_suffix(suffix_store[iz], 64, "Z", Z);

        ds[iz].dset_name  = name_store[iz];
        ds[iz].units      = "1";
        ds[iz].label      = name_store[iz];
        ds[iz].file_suffix= suffix_store[iz];
    }

    if (ion->write_total)
    {
        const size_t it = ion->nZ;
        snprintf(name_store[it], 64, "Pion_total");
        make_suffix(suffix_store[it], 64, "T", 0);

        ds[it].dset_name   = name_store[it];
        ds[it].units       = "1";
        ds[it].label       = name_store[it];
        ds[it].file_suffix = suffix_store[it];
    }

    IonGridCtx ctx;
    ctx.pulse = pulse;
    ctx.opt = ion;
    ctx.P_levels = (double *)malloc(ion->nZ * sizeof(double));
    if (!ctx.P_levels)
    {
        for (size_t k = 0; k < ndatasets; ++k) { free(suffix_store[k]); free(name_store[k]); }
        free(ds); free(suffix_store); free(name_store);
        return 7;
    }

    const int rc = dg_run_1d(req, ds, ndatasets, iongrid_eval, &ctx,
                             &ion->run, path_prefix, comm);

    free(ctx.P_levels);
    for (size_t k = 0; k < ndatasets; ++k) { free(suffix_store[k]); free(name_store[k]); }
    free(ds); free(suffix_store); free(name_store);

    return rc;
}

int iongrid_run_2d(const LaserPulse *pulse,
                   const DG_Request2D *req,
                   const IonGridOptions *ion,
                   const char *path_prefix,
                   MPI_Comm comm)
{
    if (!pulse || !req || !ion || !path_prefix) return 1;
    if (!ion->species || !ion->Z_list || ion->nZ == 0) return 2;
    if (!(ion->dt_fs > 0.0) || !(ion->tmax_fs > ion->tmin_fs)) return 3;

    const size_t ndatasets = ion->nZ + (ion->write_total ? 1u : 0u);

    DG_DatasetDesc *ds = (DG_DatasetDesc *)calloc(ndatasets, sizeof(DG_DatasetDesc));
    if (!ds) return 4;

    char **suffix_store = (char **)calloc(ndatasets, sizeof(char *));
    char **name_store   = (char **)calloc(ndatasets, sizeof(char *));
    if (!suffix_store || !name_store)
    {
        free(ds); free(suffix_store); free(name_store);
        return 5;
    }

    for (size_t i = 0; i < ndatasets; ++i)
    {
        suffix_store[i] = (char *)calloc(64, 1);
        name_store[i]   = (char *)calloc(64, 1);
        if (!suffix_store[i] || !name_store[i])
        {
            for (size_t k = 0; k <= i; ++k) { free(suffix_store[k]); free(name_store[k]); }
            free(ds); free(suffix_store); free(name_store);
            return 6;
        }
    }

    for (size_t iz = 0; iz < ion->nZ; ++iz)
    {
        const int Z = ion->Z_list[iz];
        snprintf(name_store[iz], 64, "Pion_Z%d", Z);
        make_suffix(suffix_store[iz], 64, "Z", Z);

        ds[iz].dset_name   = name_store[iz];
        ds[iz].units       = "1";
        ds[iz].label       = name_store[iz];
        ds[iz].file_suffix = suffix_store[iz];
    }

    if (ion->write_total)
    {
        const size_t it = ion->nZ;
        snprintf(name_store[it], 64, "Pion_total");
        make_suffix(suffix_store[it], 64, "T", 0);

        ds[it].dset_name   = name_store[it];
        ds[it].units       = "1";
        ds[it].label       = name_store[it];
        ds[it].file_suffix = suffix_store[it];
    }

    IonGridCtx ctx;
    ctx.pulse = pulse;
    ctx.opt = ion;
    ctx.P_levels = (double *)malloc(ion->nZ * sizeof(double));
    if (!ctx.P_levels)
    {
        for (size_t k = 0; k < ndatasets; ++k) { free(suffix_store[k]); free(name_store[k]); }
        free(ds); free(suffix_store); free(name_store);
        return 7;
    }

    const int rc = dg_run_2d(req, ds, ndatasets, iongrid_eval, &ctx,
                             &ion->run, path_prefix, comm);

    free(ctx.P_levels);
    for (size_t k = 0; k < ndatasets; ++k) { free(suffix_store[k]); free(name_store[k]); }
    free(ds); free(suffix_store); free(name_store);

    return rc;
}

