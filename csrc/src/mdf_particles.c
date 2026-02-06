#include "mdf_particles.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <hdf5.h>

#include "inputdeck.h"
#include "ionization.h"
#include "mdf.h"
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

static int stochastic_round(double x, unsigned long long *rng)
{
    if (!(x > 0.0))
        return 0;
    double f = floor(x);
    double r = x - f;
    int n = (int)f;
    if (rng_uniform01(rng) < r)
        n += 1;
    return n;
}

static int validate_opt(const MDFParticlesOptions *o)
{
    if (!o)
        return 1;
    if (!o->species || !o->Z_list || o->nZ == 0)
        return 2;

    if (!(o->tmax_fs > o->tmin_fs))
        return 3;
    if (!(o->dt_fs > 0.0))
        return 4;

    if (o->nx <= 0 || o->ny <= 0 || o->nz <= 0)
        return 5;

    /* Only require an extent if we have more than 1 cell along that axis */
    if (o->nx > 1 && !(o->xmax > o->xmin))
        return 6;
    if (o->ny > 1 && !(o->ymax > o->ymin))
        return 6;
    if (o->nz > 1 && !(o->zmax > o->zmin))
        return 6;

    if (o->ppc <= 0)
        return 7;
    if (!o->dataset_name || !o->file_suffix)
        return 8;

    return 0;
}

static void cell_bounds(double vmin, double vmax, int n, int i, double *a, double *b)
{
    if (n <= 1)
    {
        *a = vmin;
        *b = vmax;
        return;
    }
    const double dv = (vmax - vmin) / (double)n;
    *a = vmin + (double)i * dv;
    *b = *a + dv;
}

static void cell_center(double vmin, double vmax, int n, int i, double *c)
{
    if (n <= 1)
    {
        *c = 0.5 * (vmin + vmax);
        return;
    }
    double a, b;
    cell_bounds(vmin, vmax, n, i, &a, &b);
    *c = 0.5 * (a + b);
}

static size_t sample_index_from_cdf(const double *cdf, size_t n, double u_scaled)
{
    /* Find smallest i with cdf[i] >= u_scaled (binary search). */
    if (n == 0)
        return 0;
    size_t lo = 0, hi = n - 1;
    while (lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        if (cdf[mid] >= u_scaled)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

static int passes_envelope_cut(const double *Eabs, size_t Nt, double envelope_cut)
{
    if (!(envelope_cut > 0.0))
        return 1;
    double mx = 0.0;
    for (size_t i = 0; i < Nt; ++i)
        if (Eabs[i] > mx)
            mx = Eabs[i];
    return (mx >= envelope_cut) ? 1 : 0;
}

static void sim_spatial_domain(const InputSimSpec *sim,
                               double *xmin, double *xmax,
                               double *ymin, double *ymax,
                               double *zmin, double *zmax)
{
    /* Start from fixed coordinates (degenerate extent) */
    double x0 = sim->grid.fixed_x, x1 = sim->grid.fixed_x;
    double y0 = sim->grid.fixed_y, y1 = sim->grid.fixed_y;
    double z0 = sim->grid.fixed_z, z1 = sim->grid.fixed_z;

    /* Axis 1 */
    if (sim->grid.ax1 == AXIS_X)
    {
        x0 = sim->grid.ax1_min;
        x1 = sim->grid.ax1_max;
    }
    if (sim->grid.ax1 == AXIS_Y)
    {
        y0 = sim->grid.ax1_min;
        y1 = sim->grid.ax1_max;
    }
    if (sim->grid.ax1 == AXIS_Z)
    {
        z0 = sim->grid.ax1_min;
        z1 = sim->grid.ax1_max;
    }

    /* Axis 2 (if present) */
    if (sim->grid.has_ax2)
    {
        if (sim->grid.ax2 == AXIS_X)
        {
            x0 = sim->grid.ax2_min;
            x1 = sim->grid.ax2_max;
        }
        if (sim->grid.ax2 == AXIS_Y)
        {
            y0 = sim->grid.ax2_min;
            y1 = sim->grid.ax2_max;
        }
        if (sim->grid.ax2 == AXIS_Z)
        {
            z0 = sim->grid.ax2_min;
            z1 = sim->grid.ax2_max;
        }
    }

    /* Ensure min<=max even if inputdeck ever swaps them */
    if (x1 < x0)
    {
        double t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0)
    {
        double t = y0;
        y0 = y1;
        y1 = t;
    }
    if (z1 < z0)
    {
        double t = z0;
        z0 = z1;
        z1 = t;
    }

    *xmin = x0;
    *xmax = x1;
    *ymin = y0;
    *ymax = y1;
    *zmin = z0;
    *zmax = z1;
}

/* ---- dimension semantics inference (same idea as mdf_diag) ---- */

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

/* ---------------- legacy entry point (kept) ----------------
   This is your original MDF_build-based generator.
   It remains here so you don't break linkage if something still calls it.
   (Not modified.) */

int mdf_particles_run(const LaserPulse *pulse,
                      const MDFParticlesOptions *opt,
                      const char *path_prefix,
                      MPI_Comm comm)
{
    if (!pulse || !opt || !path_prefix)
        return 1;

    const int vrc = validate_opt(opt);
    if (vrc != 0)
        return 10 + vrc;

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
        free(x);
        free(y);
        free(z);
        free(px);
        free(py);
        free(pz);
        return 100;
    }

    unsigned long long rng = opt->seed;
    if (rng == 0ULL)
        rng = 0x12345678abcdefULL ^ (unsigned long long)(rank + 1) * 0x9e3779b97f4a7c15ULL;

    size_t out_k = 0;

    MDF m_cell;
    int have_m_cell = 0;
    double *cdf = NULL;
    double cdf_sum = 0.0;

    for (long long cell = c0; cell < c1; ++cell)
    {
        long long tmp = cell;
        const int ix = (int)(tmp % opt->nx);
        tmp /= opt->nx;
        const int iy = (int)(tmp % opt->ny);
        tmp /= opt->ny;
        const int iz = (int)(tmp % opt->nz);

        double xa, xb, ya, yb, za, zb;
        cell_bounds(opt->xmin, opt->xmax, opt->nx, ix, &xa, &xb);
        cell_bounds(opt->ymin, opt->ymax, opt->ny, iy, &ya, &yb);
        cell_bounds(opt->zmin, opt->zmax, opt->nz, iz, &za, &zb);

        if (!opt->mdf_at_particle_position)
        {
            double xc, yc, zc;
            cell_center(opt->xmin, opt->xmax, opt->nx, ix, &xc);
            cell_center(opt->ymin, opt->ymax, opt->ny, iy, &yc);
            cell_center(opt->zmin, opt->zmax, opt->nz, iz, &zc);

            const double r_um[3] = {xc, yc, zc};

            if (have_m_cell)
                MDF_destroy(&m_cell);
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
                continue;
            }
            have_m_cell = 1;

            free(cdf);
            cdf = (double *)malloc(m_cell.N * sizeof(double));
            if (!cdf)
            {
                MDF_destroy(&m_cell);
                have_m_cell = 0;
                free(x);
                free(y);
                free(z);
                free(px);
                free(py);
                free(pz);
                return 101;
            }

            /* Build CDF over time indices */
            cdf_sum = 0.0;
            for (size_t it = 0; it < m_cell.N; ++it)
            {
                double wsum = 0.0;
                for (size_t l = 0; l < m_cell.nZ; ++l)
                    wsum += MDF_dP(&m_cell, l, it);
                if (wsum < 0.0)
                    wsum = 0.0;
                cdf_sum += wsum;
                cdf[it] = cdf_sum;
            }

            if (!(cdf_sum > 0.0))
                continue;
        }

        for (int ip = 0; ip < opt->ppc; ++ip)
        {
            const double xr = xa + (xb - xa) * rng_uniform01(&rng);
            const double yr = ya + (yb - ya) * rng_uniform01(&rng);
            const double zr = za + (zb - za) * rng_uniform01(&rng);

            const MDF *m_use = NULL;
            const double *cdf_use = NULL;
            double sum_use = 0.0;

            MDF m_part;
            double *cdf_part = NULL;

            if (opt->mdf_at_particle_position)
            {
                const double r_um[3] = {xr, yr, zr};
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

                sum_use = 0.0;
                for (size_t it = 0; it < m_part.N; ++it)
                {
                    double wsum = 0.0;
                    for (size_t l = 0; l < m_part.nZ; ++l)
                        wsum += MDF_dP(&m_part, l, it);
                    if (wsum < 0.0)
                        wsum = 0.0;
                    sum_use += wsum;
                    cdf_part[it] = sum_use;
                }

                if (!(sum_use > 0.0))
                {
                    free(cdf_part);
                    MDF_destroy(&m_part);
                    continue;
                }

                m_use = &m_part;
                cdf_use = cdf_part;
            }
            else
            {
                m_use = &m_cell;
                cdf_use = cdf;
                sum_use = cdf_sum;
            }

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

    if (have_m_cell)
        MDF_destroy(&m_cell);
    free(cdf);

    /* Gather + write */
    const int local_count = (int)out_k;
    int *counts = NULL, *displs = NULL;

    if (rank == root)
    {
        counts = (int *)calloc((size_t)size, sizeof(int));
        displs = (int *)calloc((size_t)size, sizeof(int));
        if (!counts || !displs)
        {
            free(counts);
            free(displs);
            free(x);
            free(y);
            free(z);
            free(px);
            free(py);
            free(pz);
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

    /* Handle total==0 safely */
    float *X = NULL, *Y = NULL, *Z = NULL, *PX = NULL, *PY = NULL, *PZ = NULL;
    if (rank == root)
    {
        if (total > 0)
        {
            X = (float *)malloc((size_t)total * sizeof(float));
            Y = (float *)malloc((size_t)total * sizeof(float));
            Z = (float *)malloc((size_t)total * sizeof(float));
            PX = (float *)malloc((size_t)total * sizeof(float));
            PY = (float *)malloc((size_t)total * sizeof(float));
            PZ = (float *)malloc((size_t)total * sizeof(float));
            if (!X || !Y || !Z || !PX || !PY || !PZ)
            {
                free(X);
                free(Y);
                free(Z);
                free(PX);
                free(PY);
                free(PZ);
                free(counts);
                free(displs);
                free(x);
                free(y);
                free(z);
                free(px);
                free(py);
                free(pz);
                return 201;
            }
        }
        else
        {
            /* Dummy buffers for MPI implementations that dislike NULL recvbuf */
            X = Y = Z = PX = PY = PZ = (float *)malloc(1);
        }
    }

    MPI_Gatherv(x, local_count, MPI_FLOAT, X, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(y, local_count, MPI_FLOAT, Y, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(z, local_count, MPI_FLOAT, Z, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(px, local_count, MPI_FLOAT, PX, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(py, local_count, MPI_FLOAT, PY, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(pz, local_count, MPI_FLOAT, PZ, counts, displs, MPI_FLOAT, root, comm);

    int rcw = 0;
    if (rank == root)
    {
        if (total > 0)
        {
            DiagRequest req;
            memset(&req, 0, sizeof(req));
            req.kind = DIAG_KIND_PARTICLE;
            req.dataset_name = opt->dataset_name;
            req.time_value = 0.0;
            req.iter_value = 0;
            req.u.particle.map = (DiagParticleMap)opt->particle_map;

            DiagParticleData pdata;
            memset(&pdata, 0, sizeof(pdata));
            pdata.n = (size_t)total;
            pdata.x = X;
            pdata.y = Y;
            pdata.z = Z;
            pdata.px = PX;
            pdata.py = PY;
            pdata.pz = PZ;

            char path[4096];
            snprintf(path, sizeof(path), "%s%s", path_prefix, opt->file_suffix);
            rcw = diag_h5_write_particles(path, &req, &pdata);
        }
        /* else: nothing to write; treat as success */

        free(X);
        free(Y);
        free(Z);
        free(PX);
        free(PY);
        free(PZ);
        free(counts);
        free(displs);
    }

    MPI_Bcast(&rcw, 1, MPI_INT, root, comm);

    free(x);
    free(y);
    free(z);
    free(px);
    free(py);
    free(pz);

    return rcw;
}

/* ---------------- HDF5 cache helpers ---------------- */

static int h5_find_first_dataset(hid_t file, char name_out[128])
{
    H5G_info_t gi;
    if (H5Gget_info(file, &gi) < 0)
        return 1;

    for (hsize_t i = 0; i < gi.nlinks; ++i)
    {
        char nm[256];
        ssize_t n = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC,
                                       i, nm, sizeof(nm), H5P_DEFAULT);
        if (n <= 0)
            continue;

        H5O_info2_t oi;
        if (H5Oget_info_by_name3(file, nm, &oi, H5O_INFO_BASIC, H5P_DEFAULT) < 0)
            continue;

        if (oi.type == H5O_TYPE_DATASET)
        {
            snprintf(name_out, 128, "%s", nm);
            return 0;
        }
    }
    return 2;
}

static int h5_open_component_dataset(const char *path,
                                     const char *dset_name_guess,
                                     hid_t *file_out,
                                     hid_t *dset_out,
                                     hid_t *space_out)
{
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0)
        return 1;

    hid_t d = -1;

    if (dset_name_guess && dset_name_guess[0])
        d = H5Dopen2(f, dset_name_guess, H5P_DEFAULT);

    if (d < 0)
    {
        char first[128];
        if (h5_find_first_dataset(f, first) != 0)
        {
            H5Fclose(f);
            return 2;
        }
        d = H5Dopen2(f, first, H5P_DEFAULT);
        if (d < 0)
        {
            H5Fclose(f);
            return 3;
        }
    }

    hid_t s = H5Dget_space(d);
    if (s < 0)
    {
        H5Dclose(d);
        H5Fclose(f);
        return 4;
    }

    *file_out = f;
    *dset_out = d;
    *space_out = s;
    return 0;
}

static int clamp_index(double x, double xmin, double xmax, int n)
{
    if (n <= 1)
        return 0;
    const double dx = (xmax - xmin) / (double)(n - 1);
    if (!(dx > 0.0))
        return 0;

    const double u = (x - xmin) / dx;
    long i = lround(u);
    if (i < 0)
        i = 0;
    if (i > (long)(n - 1))
        i = (long)(n - 1);
    return (int)i;
}

static void map_r_to_spatial_indices(const InputGridSpec *g,
                                     double x, double y, double z,
                                     int *i1_out, int *i2_out)
{
    double v1 = 0.0, v2 = 0.0;

    if (g->ax1 == AXIS_X)
        v1 = x;
    else if (g->ax1 == AXIS_Y)
        v1 = y;
    else
        v1 = z;

    *i1_out = clamp_index(v1, g->ax1_min, g->ax1_max, g->ax1_n);

    if (!g->has_ax2)
    {
        *i2_out = 0;
        return;
    }

    if (g->ax2 == AXIS_X)
        v2 = x;
    else if (g->ax2 == AXIS_Y)
        v2 = y;
    else
        v2 = z;

    *i2_out = clamp_index(v2, g->ax2_min, g->ax2_max, g->ax2_n);
}

static int cache_read_component_timeseries(const InputSimSpec *sim,
                                           const char *cache_dir,
                                           const char *comp2, /* "Ex","Ax",... */
                                           int i1, int i2,
                                           double *out_t, size_t Nt)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.h5", cache_dir, comp2);

    hid_t f = -1, d = -1, s = -1;
    int rc0 = h5_open_component_dataset(path, comp2, &f, &d, &s);
    if (rc0 != 0)
        return 10 + rc0;

    const int nd = H5Sget_simple_extent_ndims(s);
    if (nd < 2 || nd > 3)
    {
        H5Sclose(s);
        H5Dclose(d);
        H5Fclose(f);
        return 20;
    }

    hsize_t dims[3] = {0, 0, 0};
    if (H5Sget_simple_extent_dims(s, dims, NULL) != nd)
    {
        H5Sclose(s);
        H5Dclose(d);
        H5Fclose(f);
        return 21;
    }

    const InputGridSpec *g = &sim->grid;

    /* Infer which dimension is time / ax1 / ax2 by matching sizes */
    char dim_sem[3] = {'?', '?', '?'};
    const int irc = infer_dim_semantics(nd, dims,
                                        (size_t)Nt,
                                        (size_t)g->ax1_n,
                                        (size_t)g->ax2_n,
                                        g->has_ax2 ? 1 : 0,
                                        dim_sem);
    if (irc != 0)
    {
        H5Sclose(s);
        H5Dclose(d);
        H5Fclose(f);
        return 22; /* could not infer semantics */
    }

    hsize_t start[3] = {0, 0, 0};
    hsize_t count[3] = {1, 1, 1};

    for (int k = 0; k < nd; ++k)
    {
        if (dim_sem[k] == 't')
        {
            start[k] = 0;
            count[k] = (hsize_t)Nt;
        }
        else if (dim_sem[k] == '1')
        {
            start[k] = (hsize_t)i1;
            count[k] = 1;
        }
        else if (dim_sem[k] == '2')
        {
            start[k] = (hsize_t)i2;
            count[k] = 1;
        }
    }

    if (H5Sselect_hyperslab(s, H5S_SELECT_SET, start, NULL, count, NULL) < 0)
    {
        H5Sclose(s);
        H5Dclose(d);
        H5Fclose(f);
        return 23;
    }

    /* Memory space is always 1D [Nt] */
    hsize_t mdims[1] = {(hsize_t)Nt};
    hid_t ms = H5Screate_simple(1, mdims, NULL);
    if (ms < 0)
    {
        H5Sclose(s);
        H5Dclose(d);
        H5Fclose(f);
        return 24;
    }

    /* Let HDF5 convert float->double if needed */
    herr_t st = H5Dread(d, H5T_NATIVE_DOUBLE, ms, s, H5P_DEFAULT, out_t);

    H5Sclose(ms);
    H5Sclose(s);
    H5Dclose(d);
    H5Fclose(f);

    return (st < 0) ? 25 : 0;
}

/* ---------------- cache-based particle generator ---------------- */

int mdf_particles_run_from_cache(const InputSimSpec *sim,
                                 const MDFParticlesOptions *opt,
                                 const char *cache_dir,
                                 const char *path_prefix,
                                 MPI_Comm comm)
{
    if (!sim || !opt || !cache_dir || !path_prefix)
        return 1;

    const int vrc = validate_opt(opt);
    if (vrc != 0)
        return 10 + vrc;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = (opt->root_rank >= 0) ? opt->root_rank : 0;

    /* --- restrict requested region to the simulation (cached-field) spatial domain --- */
    double sxmin, sxmax, symin, symax, szmin, szmax;
    sim_spatial_domain(sim, &sxmin, &sxmax, &symin, &symax, &szmin, &szmax);

    double xmin_eff = fmax(opt->xmin, sxmin);
    double xmax_eff = fmin(opt->xmax, sxmax);
    double ymin_eff = fmax(opt->ymin, symin);
    double ymax_eff = fmin(opt->ymax, symax);
    double zmin_eff = fmax(opt->zmin, szmin);
    double zmax_eff = fmin(opt->zmax, szmax);

    /* If an axis is not spatial (n==1), force it to a single coordinate */
    if (opt->nx == 1)
    {
        xmin_eff = xmax_eff = sim->grid.fixed_x;
    }
    if (opt->ny == 1)
    {
        ymin_eff = ymax_eff = sim->grid.fixed_y;
    }
    if (opt->nz == 1)
    {
        zmin_eff = zmax_eff = sim->grid.fixed_z;
    }

    /* If after intersection there is no extent on a sampled axis, nothing to generate */
    if (opt->nx > 1 && !(xmax_eff > xmin_eff))
        return 4003;
    if (opt->ny > 1 && !(ymax_eff > ymin_eff))
        return 4003;
    if (opt->nz > 1 && !(zmax_eff > zmin_eff))
        return 4003;

    /* Determine Nt robustly */
    size_t Nt = 0;
    if (sim->grid.t_n > 0)
    {
        Nt = (size_t)sim->grid.t_n;
    }
    else
    {
        if (!(sim->grid.dt > 0.0))
            return 2;
        const double span = sim->grid.t_max - sim->grid.t_min;
        if (!(span > 0.0))
            return 2;
        Nt = (size_t)llround(span / sim->grid.dt) + 1u;
    }

    if (Nt < 2)
        return 2;

    /* time array in fs (uniform) */
    double *t_fs = (double *)malloc(Nt * sizeof(double));
    if (!t_fs)
        return 3;
    for (size_t it = 0; it < Nt; ++it)
        t_fs[it] = sim->grid.t_min + (double)it * sim->grid.dt;

    const long long Ncells = (long long)opt->nx * (long long)opt->ny * (long long)opt->nz;
    const long long c0 = (Ncells * rank) / size;
    const long long c1 = (Ncells * (rank + 1)) / size;

    const long long local_cells = (c1 - c0);
    const size_t local_n = (size_t)local_cells * (size_t)opt->ppc * (size_t)opt->nZ;

    float *x = (float *)malloc(local_n * sizeof(float));
    float *y = (float *)malloc(local_n * sizeof(float));
    float *z = (float *)malloc(local_n * sizeof(float));
    float *px = (float *)malloc(local_n * sizeof(float));
    float *py = (float *)malloc(local_n * sizeof(float));
    float *pz = (float *)malloc(local_n * sizeof(float));
    if (!x || !y || !z || !px || !py || !pz)
    {
        free(t_fs);
        free(x);
        free(y);
        free(z);
        free(px);
        free(py);
        free(pz);
        return 100;
    }

    unsigned long long rng = opt->seed;
    if (rng == 0ULL)
        rng = 0x12345678abcdefULL ^ (unsigned long long)(rank + 1) * 0x9e3779b97f4a7c15ULL;

    /* Scratch buffers */
    double *Ex = (double *)malloc(Nt * sizeof(double));
    double *Ey = (double *)malloc(Nt * sizeof(double));
    double *Ez = (double *)malloc(Nt * sizeof(double));
    double *Ax = (double *)malloc(Nt * sizeof(double));
    double *Ay = (double *)malloc(Nt * sizeof(double));
    double *Az = (double *)malloc(Nt * sizeof(double));
    double *Eabs = (double *)malloc(Nt * sizeof(double));
    double *cdf = (double *)malloc(Nt * sizeof(double));

    double *w = (double *)malloc(opt->nZ * Nt * sizeof(double));
    double *S = (double *)malloc(opt->nZ * Nt * sizeof(double));
    double *dP = (double *)malloc(opt->nZ * Nt * sizeof(double));
    double *P_levels = (double *)malloc(opt->nZ * sizeof(double));

    if (!Ex || !Ey || !Ez || !Ax || !Ay || !Az || !Eabs || !cdf || !w || !S || !dP || !P_levels)
    {
        free(t_fs);
        free(x);
        free(y);
        free(z);
        free(px);
        free(py);
        free(pz);
        free(Ex);
        free(Ey);
        free(Ez);
        free(Ax);
        free(Ay);
        free(Az);
        free(Eabs);
        free(cdf);
        free(w);
        free(S);
        free(dP);
        free(P_levels);
        return 101;
    }

    size_t out_k = 0;

    for (long long cell = c0; cell < c1; ++cell)
    {
        long long tmp = cell;
        const int ix = (int)(tmp % opt->nx);
        tmp /= opt->nx;
        const int iy = (int)(tmp % opt->ny);
        tmp /= opt->ny;
        const int iz = (int)(tmp % opt->nz);

        double xa, xb, ya, yb, za, zb;
        cell_bounds(xmin_eff, xmax_eff, opt->nx, ix, &xa, &xb);
        cell_bounds(ymin_eff, ymax_eff, opt->ny, iy, &ya, &yb);
        cell_bounds(zmin_eff, zmax_eff, opt->nz, iz, &za, &zb);

        if (!opt->mdf_at_particle_position)
        {
            double xc, yc, zc;
            cell_center(xmin_eff, xmax_eff, opt->nx, ix, &xc);
            cell_center(ymin_eff, ymax_eff, opt->ny, iy, &yc);
            cell_center(zmin_eff, zmax_eff, opt->nz, iz, &zc);

            int i1 = 0, i2 = 0;
            map_r_to_spatial_indices(&sim->grid, xc, yc, zc, &i1, &i2);

            if (cache_read_component_timeseries(sim, cache_dir, "Ex", i1, i2, Ex, Nt) != 0)
                continue;
            if (cache_read_component_timeseries(sim, cache_dir, "Ey", i1, i2, Ey, Nt) != 0)
                continue;
            if (cache_read_component_timeseries(sim, cache_dir, "Ez", i1, i2, Ez, Nt) != 0)
                continue;

            /* A is required to produce momenta */
            if (cache_read_component_timeseries(sim, cache_dir, "Ax", i1, i2, Ax, Nt) != 0)
                continue;
            if (cache_read_component_timeseries(sim, cache_dir, "Ay", i1, i2, Ay, Nt) != 0)
                continue;
            if (cache_read_component_timeseries(sim, cache_dir, "Az", i1, i2, Az, Nt) != 0)
                continue;

            for (size_t it = 0; it < Nt; ++it)
                Eabs[it] = sqrt(Ex[it] * Ex[it] + Ey[it] * Ey[it] + Ez[it] * Ez[it]);

            if (!passes_envelope_cut(Eabs, Nt, opt->envelope_cut))
                continue;

            if (ION_compute_timeseries(opt->ion_model, opt->species,
                                       opt->Z_list, opt->nZ,
                                       t_fs, Nt, Eabs,
                                       w, S, dP, P_levels, NULL) != 0)
            {
                continue;
            }

            double sum = 0.0;
            for (size_t it = 0; it < Nt; ++it)
            {
                double ww = 0.0;
                for (size_t l = 0; l < opt->nZ; ++l)
                    ww += dP[l * Nt + it];
                if (ww < 0.0)
                    ww = 0.0;
                sum += ww;
                cdf[it] = sum;
            }
            if (!(sum > 0.0))
                continue;

            for (size_t l = 0; l < opt->nZ; ++l)
            {
                /* Build CDF for this level from dP_l(t) */
                double sum_l = 0.0;
                for (size_t it = 0; it < Nt; ++it)
                {
                    double wlt = dP[l * Nt + it];
                    if (wlt < 0.0)
                        wlt = 0.0;
                    sum_l += wlt;
                    cdf[it] = sum_l;
                }

                if (!(sum_l > 0.0))
                    continue;

                /* expected #particles for this level */
                const double expected = (double)opt->ppc * sum_l; /* if sum_l~1 => ~ppc */
                const int Nemit = stochastic_round(expected, &rng);
                if (Nemit <= 0)
                    continue;

                for (int ip = 0; ip < Nemit; ++ip)
                {
                    const double xr = xa + (xb - xa) * rng_uniform01(&rng);
                    const double yr = ya + (yb - ya) * rng_uniform01(&rng);
                    const double zr = za + (zb - za) * rng_uniform01(&rng);

                    const double u = rng_uniform01(&rng) * sum_l;
                    const size_t it = sample_index_from_cdf(cdf, Nt, u);

                    x[out_k] = (float)xr;
                    y[out_k] = (float)yr;
                    z[out_k] = (float)zr;

                    px[out_k] = (float)(MDF_CONV_A_TO_P * Ax[it]);
                    py[out_k] = (float)(MDF_CONV_A_TO_P * Ay[it]);
                    pz[out_k] = (float)(MDF_CONV_A_TO_P * Az[it]);

                    ++out_k;
                }
            }
        }
        else
        {
            for (int ip = 0; ip < opt->ppc; ++ip)
            {
                const double xr = xa + (xb - xa) * rng_uniform01(&rng);
                const double yr = ya + (yb - ya) * rng_uniform01(&rng);
                const double zr = za + (zb - za) * rng_uniform01(&rng);

                int i1 = 0, i2 = 0;
                map_r_to_spatial_indices(&sim->grid, xr, yr, zr, &i1, &i2);

                if (cache_read_component_timeseries(sim, cache_dir, "Ex", i1, i2, Ex, Nt) != 0)
                    continue;
                if (cache_read_component_timeseries(sim, cache_dir, "Ey", i1, i2, Ey, Nt) != 0)
                    continue;
                if (cache_read_component_timeseries(sim, cache_dir, "Ez", i1, i2, Ez, Nt) != 0)
                    continue;

                if (cache_read_component_timeseries(sim, cache_dir, "Ax", i1, i2, Ax, Nt) != 0)
                    continue;
                if (cache_read_component_timeseries(sim, cache_dir, "Ay", i1, i2, Ay, Nt) != 0)
                    continue;
                if (cache_read_component_timeseries(sim, cache_dir, "Az", i1, i2, Az, Nt) != 0)
                    continue;

                for (size_t it = 0; it < Nt; ++it)
                    Eabs[it] = sqrt(Ex[it] * Ex[it] + Ey[it] * Ey[it] + Ez[it] * Ez[it]);

                if (ION_compute_timeseries(opt->ion_model, opt->species,
                                           opt->Z_list, opt->nZ,
                                           t_fs, Nt, Eabs,
                                           w, S, dP, P_levels, NULL) != 0)
                {
                    continue;
                }

                double sum = 0.0;
                for (size_t it = 0; it < Nt; ++it)
                {
                    double ww = 0.0;
                    for (size_t l = 0; l < opt->nZ; ++l)
                        ww += dP[l * Nt + it];
                    if (ww < 0.0)
                        ww = 0.0;
                    sum += ww;
                    cdf[it] = sum;
                }
                if (!(sum > 0.0))
                    continue;

                const double u = rng_uniform01(&rng) * sum;
                const size_t it = sample_index_from_cdf(cdf, Nt, u);

                x[out_k] = (float)xr;
                y[out_k] = (float)yr;
                z[out_k] = (float)zr;

                px[out_k] = (float)(MDF_CONV_A_TO_P * Ax[it]);
                py[out_k] = (float)(MDF_CONV_A_TO_P * Ay[it]);
                pz[out_k] = (float)(MDF_CONV_A_TO_P * Az[it]);

                ++out_k;
            }
        }
    }

    free(t_fs);
    free(Ex);
    free(Ey);
    free(Ez);
    free(Ax);
    free(Ay);
    free(Az);
    free(Eabs);
    free(cdf);
    free(w);
    free(S);
    free(dP);
    free(P_levels);

    /* Gather + write (same safe handling of total==0 as above) */
    const int local_count = (int)out_k;
    int *counts = NULL, *displs = NULL;

    if (rank == root)
    {
        counts = (int *)calloc((size_t)size, sizeof(int));
        displs = (int *)calloc((size_t)size, sizeof(int));
        if (!counts || !displs)
        {
            free(counts);
            free(displs);
            free(x);
            free(y);
            free(z);
            free(px);
            free(py);
            free(pz);
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
        if (total > 0)
        {
            X = (float *)malloc((size_t)total * sizeof(float));
            Y = (float *)malloc((size_t)total * sizeof(float));
            Z = (float *)malloc((size_t)total * sizeof(float));
            PX = (float *)malloc((size_t)total * sizeof(float));
            PY = (float *)malloc((size_t)total * sizeof(float));
            PZ = (float *)malloc((size_t)total * sizeof(float));
            if (!X || !Y || !Z || !PX || !PY || !PZ)
            {
                free(X);
                free(Y);
                free(Z);
                free(PX);
                free(PY);
                free(PZ);
                free(counts);
                free(displs);
                free(x);
                free(y);
                free(z);
                free(px);
                free(py);
                free(pz);
                return 201;
            }
        }
        else
        {
            X = Y = Z = PX = PY = PZ = (float *)malloc(1);
        }
    }

    MPI_Gatherv(x, local_count, MPI_FLOAT, X, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(y, local_count, MPI_FLOAT, Y, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(z, local_count, MPI_FLOAT, Z, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(px, local_count, MPI_FLOAT, PX, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(py, local_count, MPI_FLOAT, PY, counts, displs, MPI_FLOAT, root, comm);
    MPI_Gatherv(pz, local_count, MPI_FLOAT, PZ, counts, displs, MPI_FLOAT, root, comm);

    int rcw = 0;
    if (rank == root)
    {
        if (total > 0)
        {
            DiagRequest req;
            memset(&req, 0, sizeof(req));
            req.kind = DIAG_KIND_PARTICLE;
            req.dataset_name = opt->dataset_name;
            req.time_value = 0.0;
            req.iter_value = 0;
            req.u.particle.map = (DiagParticleMap)opt->particle_map;

            DiagParticleData pdata;
            memset(&pdata, 0, sizeof(pdata));
            pdata.n = (size_t)total;
            pdata.x = X;
            pdata.y = Y;
            pdata.z = Z;
            pdata.px = PX;
            pdata.py = PY;
            pdata.pz = PZ;

            char outpath[4096];
            snprintf(outpath, sizeof(outpath), "%s%s", path_prefix, opt->file_suffix);
            rcw = diag_h5_write_particles(outpath, &req, &pdata);
        }

        free(X);
        free(Y);
        free(Z);
        free(PX);
        free(PY);
        free(PZ);
        free(counts);
        free(displs);
    }

    MPI_Bcast(&rcw, 1, MPI_INT, root, comm);

    free(x);
    free(y);
    free(z);
    free(px);
    free(py);
    free(pz);

    return rcw;
}
