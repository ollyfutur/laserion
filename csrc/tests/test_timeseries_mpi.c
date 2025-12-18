/* ============================================================================
 * File: tests/test_timeseries_mpi.c
 *
 * Purpose:
 *   Compute E(t) and A(t) at a fixed point r0 = (x0,y0,z0) for a single laser pulse.
 *   Parallelize over time indices with MPI to test speedup.
 *
 * Output:
 *   One HDF5 file per component:
 *     ts_Ex.h5, ts_Ey.h5, ts_Ez.h5, ts_Ax.h5, ts_Ay.h5, ts_Az.h5
 *
 * Notes:
 *   - diag_h5_write_field_1d() truncates/creates the file, so 1 component per file is ideal.
 *   - This program parallelizes in time: good speedup when nt is large.
 * ============================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"
#include "diag_h5.h"

/* Split N items across size ranks: contiguous blocks */
static void split_1d(int N, int size, int rank, int *n_local, int *i0_local,
                     int *counts, int *displs)
{
    const int base = (size > 0) ? (N / size) : 0;
    const int rem  = (size > 0) ? (N % size) : 0;

    int disp = 0;
    for (int r = 0; r < size; ++r)
    {
        const int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt;
        displs[r] = disp;
        disp += cnt;
    }

    *n_local  = counts[rank];
    *i0_local = displs[rank];
}

static void fill_linspace(double *t, int n, double tmin, double tmax)
{
    if (n <= 1)
    {
        if (n == 1) t[0] = tmin;
        return;
    }
    const double denom = (double)(n - 1);
    for (int i = 0; i < n; ++i)
    {
        const double s = (double)i / denom;
        t[i] = tmin + (tmax - tmin) * s;
    }
}

static int write_component_1d(const char *fname,
                              const char *dset_name,
                              const char *units,
                              const char *label,
                              const float *data,
                              size_t n,
                              const DiagAxis1D *axis_t)
{
    /* TIME/ITER are placeholders here; adjust if you want */
    const double time_value = 0.0;
    const int iter_value = 0;

    return diag_h5_write_field_1d(fname, dset_name, units, label,
                                  time_value, iter_value,
                                  data, n, axis_t);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ------------------- User knobs (edit as needed) ------------------- */

    /* Fixed probe point (um) */
    const double r0_um[3] = {0.0, 0.0, 0.0};

    /* Time axis (fs) */
    const double tmin_fs = -4000.0;
    const double tmax_fs = +4000.0;
    const int    nt      = 200000;   /* increase for stronger MPI scaling */

    /* Laser parameters (simple plane wave, linear pol) */
    const double E0_GVm        = 150.0;   /* peak field */
    const double lambda_um     = 10.0;    /* wavelength */
    const double tau_fs        = 1000.0;  /* Gaussian RMS duration */
    const double phase0        = 0.0;

    /* Propagation direction k (lab) */
    const double k_vec[3] = {0.0, 0.0, 1.0};

    /* Start position r_start (um) for retarded time definition */
    const double r_start[3] = {0.0, 0.0, 0.0};

    /* Enable retarded time? */
    const int use_retarded_time = 1;

    /* A integration parameters (must cover your t range) */
    const double A_tmin_fs = tmin_fs;
    const double A_tmax_fs = tmax_fs;
    const double A_dt_fs   = 0.2;   /* smaller => more accurate, slower */

    /* Output filename prefix */
    const char *prefix = "ts";

    /* ------------------------------------------------------------------- */

    /* Build pulse */
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, tau_fs);

    PlaneWaveProfile transverse;
    PlaneWaveProfile_init(&transverse);

    Polarization pol;
    if (LinearPolarization_init(&pol, k_vec, 0.0) != 0)
    {
        if (rank == 0) fprintf(stderr, "LinearPolarization_init failed\n");
        MPI_Finalize();
        return 1;
    }

    SinglePulse sp;
    if (SinglePulse_init(&sp,
                         E0_GVm,
                         lambda_um,
                         (const TemporalProfile *)&temporal,
                         (const TransverseProfile *)&transverse,
                         &pol,
                         phase0,
                         k_vec,
                         use_retarded_time,
                         r_start) != 0)
    {
        if (rank == 0) fprintf(stderr, "SinglePulse_init failed\n");
        MPI_Finalize();
        return 2;
    }

    /* Enable A(t,r) integration inside the pulse */
    SinglePulse_enable_A(&sp, A_tmin_fs, A_tmax_fs, A_dt_fs);

    const LaserPulse *pulse = (const LaserPulse *)&sp;

    /* Time array (all ranks need their local t values) */
    double *t_all = (double*)malloc(sizeof(double) * (size_t)nt);
    if (!t_all)
    {
        if (rank == 0) fprintf(stderr, "malloc(t_all) failed\n");
        MPI_Finalize();
        return 3;
    }
    fill_linspace(t_all, nt, tmin_fs, tmax_fs);

    /* Split time indices across ranks */
    int *counts = (int*)malloc(sizeof(int) * (size_t)size);
    int *displs = (int*)malloc(sizeof(int) * (size_t)size);
    if (!counts || !displs)
    {
        if (rank == 0) fprintf(stderr, "malloc(counts/displs) failed\n");
        free(t_all);
        MPI_Finalize();
        return 4;
    }

    int n_local = 0, i0_local = 0;
    split_1d(nt, size, rank, &n_local, &i0_local, counts, displs);

    /* Local component buffers */
    float *Ex_l = (float*)malloc(sizeof(float) * (size_t)n_local);
    float *Ey_l = (float*)malloc(sizeof(float) * (size_t)n_local);
    float *Ez_l = (float*)malloc(sizeof(float) * (size_t)n_local);
    float *Ax_l = (float*)malloc(sizeof(float) * (size_t)n_local);
    float *Ay_l = (float*)malloc(sizeof(float) * (size_t)n_local);
    float *Az_l = (float*)malloc(sizeof(float) * (size_t)n_local);

    if (!Ex_l || !Ey_l || !Ez_l || !Ax_l || !Ay_l || !Az_l)
    {
        if (rank == 0) fprintf(stderr, "malloc(local buffers) failed\n");
        free(Az_l); free(Ay_l); free(Ax_l);
        free(Ez_l); free(Ey_l); free(Ex_l);
        free(counts); free(displs); free(t_all);
        MPI_Finalize();
        return 5;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Compute local chunk */
    for (int il = 0; il < n_local; ++il)
    {
        const int ig = i0_local + il;
        const double tt = t_all[ig];

        double E[3] = {0.0, 0.0, 0.0};
        double A[3] = {0.0, 0.0, 0.0};

        LaserPulse_E(pulse, tt, r0_um, E);
        LaserPulse_A(pulse, tt, r0_um, A);

        Ex_l[il] = (float)E[0];
        Ey_l[il] = (float)E[1];
        Ez_l[il] = (float)E[2];

        Ax_l[il] = (float)A[0];
        Ay_l[il] = (float)A[1];
        Az_l[il] = (float)A[2];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* Root receive buffers */
    float *Ex = NULL, *Ey = NULL, *Ez = NULL, *Ax = NULL, *Ay = NULL, *Az = NULL;
    if (rank == 0)
    {
        Ex = (float*)malloc(sizeof(float) * (size_t)nt);
        Ey = (float*)malloc(sizeof(float) * (size_t)nt);
        Ez = (float*)malloc(sizeof(float) * (size_t)nt);
        Ax = (float*)malloc(sizeof(float) * (size_t)nt);
        Ay = (float*)malloc(sizeof(float) * (size_t)nt);
        Az = (float*)malloc(sizeof(float) * (size_t)nt);
        if (!Ex || !Ey || !Ez || !Ax || !Ay || !Az)
        {
            fprintf(stderr, "malloc(root buffers) failed\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
    }

    /* Gather each component */
    MPI_Gatherv(Ex_l, n_local, MPI_FLOAT, Ex, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Ey_l, n_local, MPI_FLOAT, Ey, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Ez_l, n_local, MPI_FLOAT, Ez, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(Ax_l, n_local, MPI_FLOAT, Ax, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Ay_l, n_local, MPI_FLOAT, Ay, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Az_l, n_local, MPI_FLOAT, Az, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if (rank == 0)
    {
        /* Axis metadata (diag_h5 will write /AXIS/AXIS1 = [vmin,vmax]) */
        DiagAxis1D axis_t = {
            .name = "t",
            .long_name = "t",
            .units = "fs",
            .vmin = tmin_fs,
            .vmax = tmax_fs
        };

        char fname[256];

        /* Write one component per file */
        snprintf(fname, sizeof(fname), "%s_Ex.h5", prefix);
        if (write_component_1d(fname, "Ex", "GV/m", "E_x", Ex, (size_t)nt, &axis_t) != 0)
            fprintf(stderr, "write failed: %s\n", fname);

        snprintf(fname, sizeof(fname), "%s_Ey.h5", prefix);
        if (write_component_1d(fname, "Ey", "GV/m", "E_y", Ey, (size_t)nt, &axis_t) != 0)
            fprintf(stderr, "write failed: %s\n", fname);

        snprintf(fname, sizeof(fname), "%s_Ez.h5", prefix);
        if (write_component_1d(fname, "Ez", "GV/m", "E_z", Ez, (size_t)nt, &axis_t) != 0)
            fprintf(stderr, "write failed: %s\n", fname);

        snprintf(fname, sizeof(fname), "%s_Ax.h5", prefix);
        if (write_component_1d(fname, "Ax", "a.u.", "A_x", Ax, (size_t)nt, &axis_t) != 0)
            fprintf(stderr, "write failed: %s\n", fname);

        snprintf(fname, sizeof(fname), "%s_Ay.h5", prefix);
        if (write_component_1d(fname, "Ay", "a.u.", "A_y", Ay, (size_t)nt, &axis_t) != 0)
            fprintf(stderr, "write failed: %s\n", fname);

        snprintf(fname, sizeof(fname), "%s_Az.h5", prefix);
        if (write_component_1d(fname, "Az", "a.u.", "A_z", Az, (size_t)nt, &axis_t) != 0)
            fprintf(stderr, "write failed: %s\n", fname);

        /* Timing */
        const double t_compute = t1 - t0;
        const double t_gather  = t2 - t1;

        printf("[rank0] nt=%d ranks=%d  compute=%g s  gather=%g s  total=%g s\n",
               nt, size, t_compute, t_gather, (t2 - t0));
        printf("[rank0] wrote: %s_Ex.h5 ... %s_Az.h5\n", prefix, prefix);

        free(Ex); free(Ey); free(Ez);
        free(Ax); free(Ay); free(Az);
    }

    free(Az_l); free(Ay_l); free(Ax_l);
    free(Ez_l); free(Ey_l); free(Ex_l);
    free(counts); free(displs);
    free(t_all);

    MPI_Finalize();
    return 0;
}

