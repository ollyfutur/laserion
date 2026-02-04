#include "run.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "inputdeck.h"
#include "laser_build.h"
#include "field_cache.h"
#include "field_diag.h"

static void print_field_diag_requests(const FieldDiagList *L, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0)
        return;

    if (!L || L->n == 0)
    {
        printf("run: no field diagnostics requested\n");
        return;
    }

    printf("run: %d field diagnostic(s) requested:\n", L->n);

    for (int i = 0; i < L->n; ++i)
    {
        const FieldDiagSpec *d = &L->v[i];

        printf("  field_diag[%d]:\n", i);
        printf("    axes = \"%s\"\n", d->axes);

        printf("    components =");
        for (int c = 0; c < d->ncomp; ++c)
            printf(" %s", d->comp[c]);
        printf("\n");

        printf("    fixed positions:");
        printf(" t=%g", d->pos_t);
        printf(" x=%g", d->pos_x);
        printf(" y=%g", d->pos_y);
        printf(" z=%g", d->pos_z);
        printf("\n");
    }

    fflush(stdout);
}

static int ensure_dir_exists_one(const char *path, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0)
    {
        if (mkdir(path, 0777) != 0)
        {
            if (errno != EEXIST)
            {
                fprintf(stderr, "run: mkdir('%s') failed: %s\n", path, strerror(errno));
                return 9001;
            }
        }
    }

    MPI_Barrier(comm);
    return 0;
}

/* mkdir -p for relative paths like "out/run1" */
static int ensure_dir_exists_p(const char *path, MPI_Comm comm)
{
    if (!path || !path[0])
        return 9002;

    /* Fast path: "." */
    if (strcmp(path, ".") == 0)
        return 0;

    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);

    /* Strip trailing slashes */
    size_t n = strlen(tmp);
    while (n > 1 && tmp[n - 1] == '/')
    {
        tmp[n - 1] = '\0';
        --n;
    }

    for (char *p = tmp + 1; *p; ++p)
    {
        if (*p == '/')
        {
            *p = '\0';
            int rc = ensure_dir_exists_one(tmp, comm);
            *p = '/';
            if (rc != 0)
                return rc;
        }
    }
    return ensure_dir_exists_one(tmp, comm);
}

static int file_exists(const char *path)
{
    struct stat st;
    return (stat(path, &st) == 0);
}

static int cache_is_complete(const char *cache_dir, int compute_A)
{
    char p[512];

    /* E components */
    snprintf(p, sizeof(p), "%s/Ex.h5", cache_dir);
    if (!file_exists(p))
        return 0;
    snprintf(p, sizeof(p), "%s/Ey.h5", cache_dir);
    if (!file_exists(p))
        return 0;
    snprintf(p, sizeof(p), "%s/Ez.h5", cache_dir);
    if (!file_exists(p))
        return 0;

    if (compute_A)
    {
        snprintf(p, sizeof(p), "%s/Ax.h5", cache_dir);
        if (!file_exists(p))
            return 0;
        snprintf(p, sizeof(p), "%s/Ay.h5", cache_dir);
        if (!file_exists(p))
            return 0;
        snprintf(p, sizeof(p), "%s/Az.h5", cache_dir);
        if (!file_exists(p))
            return 0;
    }

    return 1;
}

int run_from_inputdeck(const char *toml_path, MPI_Comm comm)
{
    InputSimSpec sim;
    int rc = inputdeck_read(toml_path, &sim);
    if (rc != 0)
        return rc;

    BuiltLasers bl;
    int rc2 = BuiltLasers_build(&sim, &bl);
    if (rc2 != 0)
    {
        inputdeck_free(&sim);
        return rc2;
    }

    const LaserPulse *pulse = BuiltLasers_active(&bl);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0)
    {
        printf("run: working_dir            = \"%s\"\n", sim.run.working_dir);
        printf("run: gas species            = \"%s\"\n", sim.run.gas);
        printf("run: ionization model       = \"%s\"\n", sim.run.ionization_model);
        printf("\n");
        fflush(stdout);
    }

    /* ------------------------- apply [run] working_dir ------------------------- */
    if (strcmp(sim.run.working_dir, ".") != 0)
    {
        int rcd0 = ensure_dir_exists_p(sim.run.working_dir, comm);
        if (rcd0 != 0)
        {
            BuiltLasers_free(&bl);
            inputdeck_free(&sim);
            return rcd0;
        }

        /* Enter working directory so everything becomes relative to it */
        if (chdir(sim.run.working_dir) != 0)
        {
            if (rank == 0)
                fprintf(stderr, "run: chdir('%s') failed: %s\n",
                        sim.run.working_dir, strerror(errno));

            BuiltLasers_free(&bl);
            inputdeck_free(&sim);
            return 9003;
        }
    }
    /* ------------------------- output layout: MS/ ------------------------- */
    {
        int rcd = 0;

        rcd = ensure_dir_exists_p("MS", comm);
        if (rcd != 0)
        {
            BuiltLasers_free(&bl);
            inputdeck_free(&sim);
            return rcd;
        }

        rcd = ensure_dir_exists_p("MS/cache", comm);
        if (rcd != 0)
        {
            BuiltLasers_free(&bl);
            inputdeck_free(&sim);
            return rcd;
        }

        rcd = ensure_dir_exists_p("MS/field", comm);
        if (rcd != 0)
        {
            BuiltLasers_free(&bl);
            inputdeck_free(&sim);
            return rcd;
        }
    }

    /* Cache directory is always "./MS/cache" inside the working directory */
    const char *cache_dir = "MS/cache";

    /* Keep sim.field_cache.out_dir consistent with the actual cache directory used. */
    snprintf(sim.field_cache.out_dir, sizeof(sim.field_cache.out_dir), "%s", cache_dir);

    /* ------------------------- gate field_cache based on inputdeck ------------ */
    const FieldCacheSpec *fc = &sim.field_cache;

    int rc3 = 0;

    if (fc->mode == FC_MODE_OFF)
    {
        /* If field diagnostics were requested, cache is required (by design). */
        if (sim.field_diag.n > 0)
        {
            if (rank == 0)
                fprintf(stderr, "run: field_diag requested but [field_cache] mode=off. "
                                "Enable cache (auto/compute/load).\n");
            rc3 = 1101;
        }
        else
        {
            rc3 = 0;
        }
    }
    else if (fc->mode == FC_MODE_LOAD)
    {
        /* Not implemented yet in field_cache.c */
        if (rank == 0)
        {
            fprintf(stderr, "run: field_cache mode=load requested, but load-path is not implemented yet.\n");
        }
        rc3 = 1001; /* project-specific error code */
    }
    else
    {
        /* AUTO / COMPUTE: compute path (or skip if write=false and no in-mem path) */
        FieldCacheOptions opt = field_cache_default_options();

        /*
         * For now, field_cache_run always writes files.
         * If fc->write==false and keep_in_memory==false, compute has no effect.
         */
        if (!fc->write && !fc->keep_in_memory)
        {
            /* Treat as off for now */
            rc3 = 0;
        }
        else if (!fc->write)
        {
            /*
             * You explicitly asked to not write cache, and keep_in_memory is true.
             * But field_diag currently reads from disk cache, so it can't run in that mode.
             */
            if (sim.field_diag.n > 0)
            {
                if (rank == 0)
                    fprintf(stderr,
                            "run: field_diag requested but field_cache.write=false; "
                            "field_diag reads cache from disk. Set field_cache.write=true for now.\n");
                rc3 = 1102;
            }
            else
            {
                rc3 = 0;
            }
        }
        else
        {
            int rcd = ensure_dir_exists_p(cache_dir, comm);
            if (rcd != 0)
            {
                BuiltLasers_free(&bl);
                inputdeck_free(&sim);
                return rcd;
            }

            int have_cache = 0;
            if (rank == opt.root_rank)
                have_cache = cache_is_complete(cache_dir, opt.compute_A);

            MPI_Bcast(&have_cache, 1, MPI_INT, opt.root_rank, comm);

            switch (fc->mode)
            {
            case FC_MODE_OFF:
                rc3 = 0;
                break;

            case FC_MODE_LOAD:
                if (!have_cache)
                {
                    if (rank == opt.root_rank)
                        fprintf(stderr,
                                "run: field_cache mode=load but cache is missing in %s/\n",
                                cache_dir);
                    rc3 = 1002;
                }
                else
                {
                    if (rank == opt.root_rank)
                    {
                        printf("run: field_cache load successful (cache found in %s/)\n",
                               cache_dir);
                        fflush(stdout);
                    }
                    rc3 = 0;
                }
                break;

            case FC_MODE_AUTO:
                if (have_cache)
                {
                    if (rank == opt.root_rank)
                    {
                        printf("run: field_cache auto mode — using existing cache in %s/\n",
                               cache_dir);
                        fflush(stdout);
                    }
                    rc3 = 0;
                }
                else
                {
                    if (rank == opt.root_rank)
                    {
                        printf("run: field_cache auto mode — no cache found, computing cache\n");
                        fflush(stdout);
                    }
                    rc3 = field_cache_run(&sim, pulse, cache_dir, &opt, comm);
                }
                break;

            case FC_MODE_COMPUTE:
                if (rank == opt.root_rank)
                {
                    printf("run: field_cache compute mode — recomputing cache in %s/\n",
                           cache_dir);
                    fflush(stdout);
                }
                rc3 = field_cache_run(&sim, pulse, cache_dir, &opt, comm);
                break;
            }
        }
    }

    /* ------------------------- run field diagnostics from cache --------------- */
    if (rc3 == 0 && sim.field_diag.n > 0)
    {
        /* Outputs go to ./MS (inside working_dir). */
        int rcd = ensure_dir_exists_p("MS", comm);
        if (rcd != 0)
        {
            BuiltLasers_free(&bl);
            inputdeck_free(&sim);
            return rcd;
        }

        FieldDiagOptions fdopt = field_diag_default_options();
        fdopt.root_rank = 0;
        fdopt.write_time_iter_0 = 1;

        if (rank == fdopt.root_rank)
        {
            printf("run: field_diag — writing %d diagnostic(s) to diag/ (prefix diag/fields)\n",
                   sim.field_diag.n);
            print_field_diag_requests(&sim.field_diag, comm);
            fflush(stdout);
        }
        printf("\n");
        int fdr = field_diag_run_from_cache(&sim, "MS/field", &fdopt, comm);
        if (fdr != 0 && rc3 == 0)
            rc3 = 1200 + fdr;
    }

    printf("run: done!\n");

    BuiltLasers_free(&bl);
    inputdeck_free(&sim);
    return rc3;
}
