#include "driver.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "inputdeck.h"
#include "laser_build.h"
#include "field_cache.h"

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
                fprintf(stderr, "driver: mkdir('%s') failed: %s\n", path, strerror(errno));
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

    /* Gate field_cache based on inputdeck */
    const FieldCacheSpec *fc = &sim.field_cache;

    int rc3 = 0;
    if (fc->mode == FC_MODE_OFF)
    {
        /* Nothing to do */
        rc3 = 0;
    }
    else if (fc->mode == FC_MODE_LOAD)
    {
        /* Not implemented yet in field_cache.c */
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0)
        {
            fprintf(stderr, "driver: field_cache mode=load requested, but load-path is not implemented yet.\n");
        }
        rc3 = 1001; /* choose a project-specific error code */
    }
    else
    {
        /* AUTO / COMPUTE: compute path */
        FieldCacheOptions opt = field_cache_default_options();

        /*
         * For now, field_cache_run always writes files.
         * If fc->write==false is requested, you either:
         *  - treat it as OFF (skip), OR
         *  - implement a no-write path in field_cache.c later.
         *
         * Here: skip if write=false (most honest behavior until keep_in_memory exists).
         */
        if (!fc->write)
        {
            rc3 = 0;
        }
        else
        {
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
                    int rank = 0;
                    MPI_Comm_rank(comm, &rank);
                    if (rank == 0)
                        fprintf(stderr, "driver: chdir('%s') failed: %s\n", sim.run.working_dir, strerror(errno));

                    BuiltLasers_free(&bl);
                    inputdeck_free(&sim);
                    return 9003;
                }
            }

            /* Cache directory is always "./cache" inside the working directory */
            const char *cache_dir = "cache";
            int rcd = ensure_dir_exists_p(cache_dir, comm);
            if (rcd != 0)
            {
                BuiltLasers_free(&bl);
                inputdeck_free(&sim);
                return rcd;
            }

            /* ------------------------- field_cache mode logic -------------------------- */
            const FieldCacheSpec *fc = &sim.field_cache;

            /* If we are not writing and not keeping in memory, compute has no effect currently */
            if (!fc->write && !fc->keep_in_memory)
            {
                /* Treat as off for now */
                BuiltLasers_free(&bl);
                inputdeck_free(&sim);
                return 0;
            }

            int rank = 0;
            MPI_Comm_rank(comm, &rank);

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
                                "driver: field_cache mode=load but cache is missing in %s/\n",
                                cache_dir);
                    rc3 = 1002;
                }
                else
                {
                    if (rank == opt.root_rank)
                    {
                        printf("driver: field_cache load successful (cache found in %s/)\n",
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
                        printf("driver: field_cache auto mode — using existing cache in %s/\n",
                               cache_dir);
                        fflush(stdout);
                    }
                    rc3 = 0;
                }
                else
                {
                    if (rank == opt.root_rank)
                    {
                        printf("driver: field_cache auto mode — no cache found, computing cache\n");
                        fflush(stdout);
                    }
                    rc3 = field_cache_run(&sim, pulse, cache_dir, &opt, comm);
                }
                break;
            case FC_MODE_COMPUTE:
                if (rank == opt.root_rank)
                {
                    printf("driver: field_cache compute mode — recomputing cache in %s/\n",
                           cache_dir);
                    fflush(stdout);
                }
                rc3 = field_cache_run(&sim, pulse, cache_dir, &opt, comm);
                break;
            }
        }
    }

    BuiltLasers_free(&bl);
    inputdeck_free(&sim);
    return rc3;
}
