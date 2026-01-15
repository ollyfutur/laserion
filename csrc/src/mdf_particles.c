#include "mdf_particles.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "diag_h5.h"

/* ---------------- RNG: splitmix64 + uniform(0,1) ---------------- */

static unsigned long long splitmix64_next(unsigned long long *state)
{
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rng_uniform01(unsigned long long *state)
{
    /* 53-bit mantissa uniform in [0,1) */
    const unsigned long long r = splitmix64_next(state);
    return (double)((r >> 11) & 0x1fffffffffffffULL) * (1.0 / 9007199254740992.0);
}

/* ---------------- utilities ---------------- */

static int validate_opt(const MDFParticlesOptions *o)
{
    if (!o) return 1;
    if (!o->species || !o->Z_list || o->nZ == 0) return 2;

    if (!(o->tmax_fs > o->tmin_fs)) return 3;
    if (!(o->dt_fs > 0.0)) return 4;

    if (o->nx <= 0 || o->ny <= 0 || o->nz <= 0) return 5;
    if (!(o->xmax > o->xmin) || !(o->ymax > o->ymin) || !(o->zmax > o->zmin)) return 6;

    if (o->ppc <= 0) return 7;

    if (!o->dataset_name || !o->file_suffix) return 8;

    return 0;
}

static void cell_bounds(double vmin, double vmax, int n, int i, double *a, double *b)
{
    const double dv = (vmax - vmin) / (double)n;
    *a = vmin + (double)i * dv;
    *b = *a + dv;
}

static void cell_center(double vmin, double vmax, int n, int i, double *c)
{
    double a, b;
    cell_bounds(vmin, vmax, n, i, &a, &b);
    *c = 0.5 * (a + b);
}

/* Build CDF over time indices: w_it = sum_l MDF_dP(l,it). Returns total weight. */
static double build_time_cdf(const MDF *m, double *cdf /* size m->N */)
{
    double sum = 0.0;
    for (size_t it = 0; it < m->N; ++it)
    {
        double w = 0.0;
        for (size_t l = 0; l < m->nZ; ++l)
            w += MDF_dP(m, l, it);

        if (w < 0.0) w = 0.0;
        sum += w;
        cdf[it] = sum;
    }
    return sum;
}

static size_t sample_index_from_cdf(const double *cdf, size_t n, double u_scaled)
{
    /* Find smallest i with cdf[i] >= u_scaled (binary search). */
    size_t lo = 0, hi = (n == 0) ? 0 : (n - 1);
    while (lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        if (cdf[mid] >= u_scaled) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

/* ---------------- main entry point ---------------- */

int mdf_particles_run(const LaserPulse *pulse,
                      const MDFParticlesOptions *opt,
                      const char *path_prefix,
                      MPI_Comm comm)
{
    if (!pulse || !opt || !path_prefix) return 1;

    const int vrc = validate_opt(opt);
    if (vrc != 0) return 10 + vrc;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = (opt->root_rank >= 0) ? opt->root_rank : 0;

    const long long Ncells = (long long)opt->nx * (long long)opt->ny * (long long)opt->nz;
    const long long c0 = (Ncells * rank) / size;
    const long long c1 = (Ncells * (rank + 1)) / size;

    const long long local_cells = (c1 - c0);
    const size_t local_n = (size_t)local_cells * (size_t)opt->ppc;

    float *x = (float *)malloc(local_n * sizeof(float));
    float *y = (float *)malloc(local_n * sizeof(float));
    float *z = (float *)malloc(local_n * sizeof(float));
    float *px = (float *)malloc(local_n * sizeof(float));
    float *py = (float *)malloc(local_n * sizeof(float));
    float *pz = (float *)malloc(local_n * sizeof(float));
    if (!x || !y || !z || !px || !py || !pz)
    {
        free(x); free(y); free(z); free(px); free(py); free(pz);
        return 100;
    }

    unsigned long long rng = opt->seed;
    if (rng == 0ULL)
        rng = 0x12345678abcdefULL ^ (unsigned long long)(rank + 1) * 0x9e3779b97f4a7c15ULL;

    size_t out_k = 0;

    /* Reusable MDF storage if we compute per-cell MDF at center. */
    MDF m_cell;
    int have_m_cell = 0;
    double *cdf = NULL;
    double cdf_sum = 0.0;

    for (long long cell = c0; cell < c1; ++cell)
    {
        long long tmp = cell;
        const int ix = (int)(tmp % opt->nx); tmp /= opt->nx;
        const int iy = (int)(tmp % opt->ny); tmp /= opt->ny;
        const int iz = (int)(tmp % opt->nz);

        double xa, xb, ya, yb, za, zb;
        cell_bounds(opt->xmin, opt->xmax, opt->nx, ix, &xa, &xb);
        cell_bounds(opt->ymin, opt->ymax, opt->ny, iy, &ya, &yb);
        cell_bounds(opt->zmin, opt->zmax, opt->nz, iz, &za, &zb);

        /* If we sample MDF once per cell, build at center now. */
        if (!opt->mdf_at_particle_position)
        {
            double xc, yc, zc;
            cell_center(opt->xmin, opt->xmax, opt->nx, ix, &xc);
            cell_center(opt->ymin, opt->ymax, opt->ny, iy, &yc);
            cell_center(opt->zmin, opt->zmax, opt->nz, iz, &zc);

            const double r_um[3] = { xc, yc, zc };

            if (have_m_cell) MDF_destroy(&m_cell);
            have_m_cell = 0;

            if (MDF_build(&m_cell,
                          pulse,
                          opt->species,
                          opt->Z_list, opt->nZ,
                          r_um,
                          opt->ion_model,
                          opt->envelope_cut,
                          opt->tmin_fs, opt->tmax_fs, opt->dt_fs) != 0)
            {
                /* If MDF build fails for this cell, skip generating particles for it. */
                continue;
            }
            have_m_cell = 1;

            free(cdf);
            cdf = (double *)malloc(m_cell.N * sizeof(double));
            if (!cdf)
            {
                MDF_destroy(&m_cell);
                have_m_cell = 0;
                free(x); free(y); free(z); free(px); free(py); free(pz);
                return 101;
            }

            cdf_sum = build_time_cdf(&m_cell, cdf);
            if (!(cdf_sum > 0.0))
            {
                /* No ionization weight -> skip cell */
                continue;
            }
        }

        /* Generate ppc particles for this cell. */
        for (int ip = 0; ip < opt->ppc; ++ip)
        {
            /* Sample position uniformly inside cell. */
            const double xr = xa + (xb - xa) * rng_uniform01(&rng);
            const double yr = ya + (yb - ya) * rng_uniform01(&rng);
            const double zr = za + (zb - za) * rng_uniform01(&rng);

            MDF m_part;
            double *cdf_part = NULL;
            double sum_part = 0.0;

            const MDF *m_use = NULL;
            const double *cdf_use = NULL;
            double sum_use = 0.0;

            if (opt->mdf_at_particle_position)
            {
                const double r_um[3] = { xr, yr, zr };

                if (MDF_build(&m_part,
                              pulse,
                              opt->species,
                              opt->Z_list, opt->nZ,
                              r_um,
                              opt->ion_model,
                              opt->envelope_cut,
                              opt->tmin_fs, opt->tmax_fs, opt->dt_fs) != 0)
                {
                    continue;
                }

                cdf_part = (double *)malloc(m_part.N * sizeof(double));
                if (!cdf_part)
                {
                    MDF_destroy(&m_part);
                    continue;
                }

                sum_part = build_time_cdf(&m_part, cdf_part);
                if (!(sum_part > 0.0))
                {
                    free(cdf_part);
                    MDF_destroy(&m_part);
                    continue;
                }

                m_use = &m_part;
                cdf_use = cdf_part;
                sum_use = sum_part;
            }
            else
            {
                m_use = &m_cell;
                cdf_use = cdf;
                sum_use = cdf_sum;
            }

            /* Sample time index from CDF and take associated momentum. */
            const double u = rng_uniform01(&rng) * sum_use;
            const size_t it = sample_index_from_cdf(cdf_use, m_use->N, u);

            x[out_k] = (float)xr;
            y[out_k] = (float)yr;
            z[out_k] = (float)zr;

            px[out_k] = (float)m_use->p[it][0];
            py[out_k] = (float)m_use->p[it][1];
            pz[out_k] = (float)m_use->p[it][2];

            ++out_k;

            if (opt->mdf_at_particle_position)
            {
                free(cdf_part);
                MDF_destroy(&m_part);
            }
        }
    }

    if (have_m_cell) MDF_destroy(&m_cell);
    free(cdf);

    /* out_k may be <= local_n if we skipped cells/particles. */
    const int local_count = (int)out_k;
    int *counts = NULL;
    int *displs = NULL;

    if (rank == root)
    {
        counts = (int *)calloc((size_t)size, sizeof(int));
        displs = (int *)calloc((size_t)size, sizeof(int));
        if (!counts || !displs)
        {
            free(counts); free(displs);
            free(x); free(y); free(z); free(px); free(py); free(pz);
            return 200;
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, root, comm);

    int total = 0;
    if (rank == root)
    {
        for (int r = 0; r < size; ++r)
        {
            displs[r] = total;
            total += counts[r];
        }
    }

    float *X = NULL, *Y = NULL, *Z = NULL, *PX = NULL, *PY = NULL, *PZ = NULL;

    if (rank == root)
    {
        X  = (float *)malloc((size_t)total * sizeof(float));
        Y  = (float *)malloc((size_t)total * sizeof(float));
        Z  = (float *)malloc((size_t)total * sizeof(float));
        PX = (float *)malloc((size_t)total * sizeof(float));
        PY = (float *)malloc((size_t)total * sizeof(float));
        PZ = (float *)malloc((size_t)total * sizeof(float));
        if (!X || !Y || !Z || !PX || !PY || !PZ)
        {
            free(X); free(Y); free(Z); free(PX); free(PY); free(PZ);
            free(counts); free(displs);
            free(x); free(y); free(z); free(px); free(py); free(pz);
            return 201;
        }
    }

    MPI_Gatherv(x,  local_count, MPI_FLOAT, X,  counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(y,  local_count, MPI_FLOAT, Y,  counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(z,  local_count, MPI_FLOAT, Z,  counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(px, local_count, MPI_FLOAT, PX, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(py, local_count, MPI_FLOAT, PY, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(pz, local_count, MPI_FLOAT, PZ, counts, displs, MPI_FLOAT, root, comm);

    int rc = 0;

    if (rank == root)
    {
        DiagRequest req;
        memset(&req, 0, sizeof(req));

        req.kind = DIAG_KIND_PARTICLE;
        req.dataset_name = opt->dataset_name;
        req.time_value = 0.0;
        req.iter_value = 0;

        /* Use your desired mapping by default; caller can override via opt->particle_map. */
        req.u.particle.map = (DiagParticleMap)opt->particle_map;

        DiagParticleData pdata;
        memset(&pdata, 0, sizeof(pdata));
        pdata.n  = (size_t)total;
        pdata.x  = X;
        pdata.y  = Y;
        pdata.z  = Z;
        pdata.px = PX;
        pdata.py = PY;
        pdata.pz = PZ;

        char path[4096];
        snprintf(path, sizeof(path), "%s%s", path_prefix, opt->file_suffix);

        rc = diag_h5_write_particles(path, &req, &pdata);

        free(X); free(Y); free(Z);
        free(PX); free(PY); free(PZ);
        free(counts); free(displs);
    }

    MPI_Bcast(&rc, 1, MPI_INT, root, comm);

    free(x); free(y); free(z);
    free(px); free(py); free(pz);

    return rc;
}

