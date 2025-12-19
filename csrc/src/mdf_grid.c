#include "mdf_grid.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ------------------------------ utilities ------------------------------ */

static int is_2d_kind(MDFGridKind k)
{
    return (k == MDF_GRID_F_PX_PY) || (k == MDF_GRID_F_PY_PZ) || (k == MDF_GRID_F_PX_PZ);
}

static void project_p(MDFGridKind kind, const double p[3], double *p1, double *p2)
{
    const double px = p[0], py = p[1], pz = p[2];

    switch (kind)
    {
        case MDF_GRID_F_PX:     *p1 = px; *p2 = 0.0; break;
        case MDF_GRID_F_PY:     *p1 = py; *p2 = 0.0; break;
        case MDF_GRID_F_PZ:     *p1 = pz; *p2 = 0.0; break;
        case MDF_GRID_F_PX_PY:  *p1 = px; *p2 = py;  break;
        case MDF_GRID_F_PY_PZ:  *p1 = py; *p2 = pz;  break;
        case MDF_GRID_F_PX_PZ:  *p1 = px; *p2 = pz;  break;
        default:                *p1 = 0.0; *p2 = 0.0; break;
    }
}

static int clamp_bin(double v, double vmin, double vmax, int nbins)
{
    if (!(vmax > vmin) || nbins <= 0) return -1;
    if (v < vmin || v >= vmax) return -1;

    const double u = (v - vmin) / (vmax - vmin); /* [0,1) */
    int b = (int)floor(u * (double)nbins);

    if (b < 0) b = 0;
    if (b >= nbins) b = nbins - 1;
    return b;
}

static void linspace_centers(double a, double b, int n, double *out)
{
    if (n <= 0) return;
    const double da = (b - a) / (double)n;
    for (int i = 0; i < n; ++i)
        out[i] = a + (i + 0.5) * da;
}

/* ------------------------------ validation ----------------------------- */

static int validate_opt(const MDFGridOptions *o)
{
    if (!o) return 1;
    if (!o->species || !o->Z_list || o->nZ == 0) return 2;

    if (!(o->tmax_fs > o->tmin_fs)) return 3; /* MDF_build requires explicit window */

    if (o->nx <= 0 || o->ny <= 0 || o->nz <= 0) return 4;
    if (!(o->xmax > o->xmin) || !(o->ymax > o->ymin) || !(o->zmax > o->zmin)) return 5;

    if (o->nbins1 <= 0 || !(o->p1max > o->p1min)) return 6;

    if (is_2d_kind(o->kind))
    {
        if (o->nbins2 <= 0 || !(o->p2max > o->p2min)) return 7;
    }

    if (!o->dataset_name || !o->label || !o->units || !o->file_suffix) return 8;

    return 0;
}

/* ------------------------------ core logic ----------------------------- */

int mdfgrid_run(const LaserPulse *pulse,
                const MDFGridOptions *opt,
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

    const int n1 = opt->nbins1;
    const int n2 = is_2d_kind(opt->kind) ? opt->nbins2 : 1;

    const size_t hist_sz = (size_t)n1 * (size_t)n2;

    double *hist_local  = (double *)calloc(hist_sz, sizeof(double));
    double *hist_global = (double *)calloc(hist_sz, sizeof(double));
    if (!hist_local || !hist_global)
    {
        free(hist_local);
        free(hist_global);
        return 100;
    }

    /* Spatial sampling grid (centers). */
    double *xs = (double *)malloc((size_t)opt->nx * sizeof(double));
    double *ys = (double *)malloc((size_t)opt->ny * sizeof(double));
    double *zs = (double *)malloc((size_t)opt->nz * sizeof(double));
    if (!xs || !ys || !zs)
    {
        free(xs); free(ys); free(zs);
        free(hist_local); free(hist_global);
        return 101;
    }

    linspace_centers(opt->xmin, opt->xmax, opt->nx, xs);
    linspace_centers(opt->ymin, opt->ymax, opt->ny, ys);
    linspace_centers(opt->zmin, opt->zmax, opt->nz, zs);

    const long long Npts = (long long)opt->nx * (long long)opt->ny * (long long)opt->nz;
    const long long i0 = (Npts * rank) / size;
    const long long i1 = (Npts * (rank + 1)) / size;

    /* For each spatial point owned by this rank, build MDF and accumulate histogram. */
    for (long long idx = i0; idx < i1; ++idx)
    {
        long long tmp = idx;
        const int ix = (int)(tmp % opt->nx); tmp /= opt->nx;
        const int iy = (int)(tmp % opt->ny); tmp /= opt->ny;
        const int iz = (int)(tmp % opt->nz);

        const double r_um[3] = { xs[ix], ys[iy], zs[iz] };

        MDF m;
        const int mrc = MDF_build(&m,
                                 pulse,
                                 opt->species,
                                 opt->Z_list, opt->nZ,
                                 r_um,
                                 opt->ion_model,
                                 opt->envelope_cut,
                                 opt->tmin_fs,
                                 opt->tmax_fs,
                                 opt->dt_fs);
        if (mrc != 0)
        {
            /* If MDF_build fails at one point, skip that point (do not abort run). */
            continue;
        }

        /* Histogram p(t,r) weighted by dP(level,t). */
        for (size_t l = 0; l < m.nZ; ++l)
        {
            for (size_t it = 0; it < m.N; ++it)
            {
                const double w = MDF_dP(&m, l, it);
                if (!(w > 0.0)) continue;

                double p1, p2;
                project_p(opt->kind, m.p[it], &p1, &p2);

                const int b1 = clamp_bin(p1, opt->p1min, opt->p1max, n1);
                if (b1 < 0) continue;

                if (!is_2d_kind(opt->kind))
                {
                    hist_local[(size_t)b1] += w;
                }
                else
                {
                    const int b2 = clamp_bin(p2, opt->p2min, opt->p2max, n2);
                    if (b2 < 0) continue;

                    /* C-order [p2][p1] */
                    hist_local[(size_t)b2 * (size_t)n1 + (size_t)b1] += w;
                }
            }
        }

        MDF_destroy(&m);
    }

    /* Sum histograms across ranks. */
    MPI_Allreduce(hist_local, hist_global, (int)hist_sz, MPI_DOUBLE, MPI_SUM, comm);

    /* Optional normalization: sum(f)=1 */
    if (opt->normalize_sum_to_1)
    {
        double sum = 0.0;
        for (size_t i = 0; i < hist_sz; ++i) sum += hist_global[i];
        if (sum > 0.0)
            for (size_t i = 0; i < hist_sz; ++i) hist_global[i] /= sum;
    }

    /* Root writes H5 using diag_h5 grid writer (float32 payload). */
    int rc = 0;
    if (rank == root)
    {
        float *data_f32 = (float *)malloc(hist_sz * sizeof(float));
        if (!data_f32)
        {
            rc = 200;
        }
        else
        {
            for (size_t i = 0; i < hist_sz; ++i)
                data_f32[i] = (float)hist_global[i];

            char path[4096];
            snprintf(path, sizeof(path), "%s%s", path_prefix, opt->file_suffix);

            /* Use fixed coords to store the region center (purely informative). */
            DiagFixedCoords fixed;
            fixed.t = 0.0;
            fixed.x = 0.5 * (opt->xmin + opt->xmax);
            fixed.y = 0.5 * (opt->ymin + opt->ymax);
            fixed.z = 0.5 * (opt->zmin + opt->zmax);

            /* Axis metadata: we encode momentum axes with DiagAxisID X/Y and units m_e c. */
            DiagAxis ax1, ax2;
            ax1.id = DIAG_AXIS_X;
            ax1.long_name = "p1";
            ax1.units = "m_e c";
            ax1.vmin = opt->p1min;
            ax1.vmax = opt->p1max;

            if (!is_2d_kind(opt->kind))
            {
                rc = diag_h5_write_grid_1d(path,
                                           opt->dataset_name,
                                           opt->units,
                                           opt->label,
                                           0.0, 0,
                                           data_f32,
                                           (size_t)n1,
                                           &ax1,
                                           &fixed);
            }
            else
            {
                ax2.id = DIAG_AXIS_Y;
                ax2.long_name = "p2";
                ax2.units = "m_e c";
                ax2.vmin = opt->p2min;
                ax2.vmax = opt->p2max;

                rc = diag_h5_write_grid_2d(path,
                                           opt->dataset_name,
                                           opt->units,
                                           opt->label,
                                           0.0, 0,
                                           data_f32,
                                           (size_t)n1, (size_t)n2,
                                           &ax1, &ax2,
                                           &fixed);
            }

            free(data_f32);
        }
    }

    /* Share rc with all ranks. */
    MPI_Bcast(&rc, 1, MPI_INT, root, comm);

    free(xs); free(ys); free(zs);
    free(hist_local);
    free(hist_global);
    return rc;
}

