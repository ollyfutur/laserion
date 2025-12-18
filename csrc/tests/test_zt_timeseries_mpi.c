/* ============================================================================
 * File: tests/test_zt_timeseries_mpi.c
 *
 * Purpose:
 *   Compute E(z,t) and A(z,t) at fixed (x,y) = (0,0) for a single laser pulse.
 *   This is a 2D diagnostic with axes (t, z), written as 2D HDF5 per component.
 *
 * Parallelization:
 *   MPI splits the z-index (rows). Each rank computes a slab of z rows over all t.
 *   Root gathers and writes one HDF5 file per component (diag_h5_write_field_2d truncates).
 * ============================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"
#include "diag_h5.h"

static void progress_print_rank0(int done, int total, double t_start)
{
    if (total <= 0)
        return;

    const double frac = (double)done / (double)total;
    const double now = MPI_Wtime();
    const double dt = now - t_start;

    double eta = 0.0;
    if (done > 0 && frac > 0.0)
        eta = dt * (1.0 - frac) / frac;

    /* Use newline (more reliable through mpirun than '\r') */
    fprintf(stderr,
            "[progress] %d/%d (%.1f%%) elapsed=%.1fs ETA=%.1fs\n",
            done, total, 100.0 * frac, dt, eta);
    fflush(stderr);
}

/* Split N items across size ranks: contiguous blocks */
static void split_1d(int N, int size, int rank, int *n_local, int *i0_local,
                     int *counts, int *displs)
{
    const int base = (size > 0) ? (N / size) : 0;
    const int rem = (size > 0) ? (N % size) : 0;

    int disp = 0;
    for (int r = 0; r < size; ++r)
    {
        const int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt;
        displs[r] = disp;
        disp += cnt;
    }

    *n_local = counts[rank];
    *i0_local = displs[rank];
}

static void fill_linspace(double *x, int n, double xmin, double xmax)
{
    if (n <= 1)
    {
        if (n == 1)
            x[0] = xmin;
        return;
    }
    const double denom = (double)(n - 1);
    for (int i = 0; i < n; ++i)
    {
        const double s = (double)i / denom;
        x[i] = xmin + (xmax - xmin) * s;
    }
}

static int write_component_2d(const char *fname,
                              const char *dset_name,
                              const char *units,
                              const char *label,
                              const float *data, /* size = n2*n1 (z rows, t cols) */
                              size_t n1,         /* AXIS1 (fastest) = Nt */
                              size_t n2,         /* AXIS2 (slow)    = Nz */
                              const DiagAxis1D *axis1,
                              const DiagAxis1D *axis2)
{
    /* TIME/ITER placeholders */
    const double time_value = 0.0;
    const int iter_value = 0;

    return diag_h5_write_field_2d(fname,
                                  dset_name, units, label,
                                  time_value, iter_value,
                                  data,
                                  n1, n2,
                                  axis1, axis2);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    /* Make progress output appear immediately under mpirun */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ------------------- User knobs (edit as needed) ------------------- */

    /* Fixed transverse coordinates (um) */
    const double x0_um = 0.0;
    const double y0_um = 0.0;

    /* z axis (um) */
    const double zmin_um = -200.0;
    const double zmax_um = +200.0;
    const int nz = 801;

    /* time axis (fs) */
    const double tmin_fs = -4000.0;
    const double tmax_fs = +4000.0;
    const int nt = 2000; /* increase for stronger MPI scaling */

    /* Laser parameters (simple plane wave, linear pol) */
    const double E0_GVm = 150.0;
    const double lambda_um = 10.0;
    const double tau_fs = 1000.0;
    const double phase0 = 0.0;

    /* Propagation direction k (lab) */
    const double k_vec[3] = {0.0, 0.0, 1.0};

    /* Start position r_start (um) for retarded time definition */
    const double r_start[3] = {0.0, 0.0, 0.0};

    /* Enable retarded time? */
    const int use_retarded_time = 1;

    /* A integration parameters (must cover your t range) */
    const double A_tmin_fs = tmin_fs;
    const double A_tmax_fs = tmax_fs;
    const double A_dt_fs = 0.2;

    /* Output filename prefix */
    const char *prefix = "zt";

    /* ------------------------------------------------------------------- */

    /* Build pulse (same construction pattern as your timeseries MPI test) */
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, tau_fs);

    PlaneWaveProfile transverse;
    PlaneWaveProfile_init(&transverse);

    Polarization pol;
    if (LinearPolarization_init(&pol, k_vec, 0.0) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "LinearPolarization_init failed\n");
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
        if (rank == 0)
            fprintf(stderr, "SinglePulse_init failed\n");
        MPI_Finalize();
        return 2;
    }

    SinglePulse_enable_A(&sp, A_tmin_fs, A_tmax_fs, A_dt_fs);

    const LaserPulse *pulse = (const LaserPulse *)&sp;

    /* Axes arrays */
    double *t_all = (double *)malloc(sizeof(double) * (size_t)nt);
    double *z_all = (double *)malloc(sizeof(double) * (size_t)nz);
    if (!t_all || !z_all)
    {
        if (rank == 0)
            fprintf(stderr, "malloc(t_all/z_all) failed\n");
        free(t_all);
        free(z_all);
        MPI_Finalize();
        return 3;
    }
    fill_linspace(t_all, nt, tmin_fs, tmax_fs);
    fill_linspace(z_all, nz, zmin_um, zmax_um);

    /* Split z indices across ranks */
    int *z_counts = (int *)malloc(sizeof(int) * (size_t)size);
    int *z_displs = (int *)malloc(sizeof(int) * (size_t)size);
    if (!z_counts || !z_displs)
    {
        if (rank == 0)
            fprintf(stderr, "malloc(z_counts/z_displs) failed\n");
        free(t_all);
        free(z_all);
        free(z_counts);
        free(z_displs);
        MPI_Finalize();
        return 4;
    }

    int nz_local = 0, iz0_local = 0;
    split_1d(nz, size, rank, &nz_local, &iz0_local, z_counts, z_displs);

    /* Local arrays for each component: layout [z_local][t] with t fastest */
    const size_t n_local = (size_t)nz_local * (size_t)nt;

    float *Ex_l = (float *)malloc(sizeof(float) * n_local);
    float *Ey_l = (float *)malloc(sizeof(float) * n_local);
    float *Ez_l = (float *)malloc(sizeof(float) * n_local);
    float *Ax_l = (float *)malloc(sizeof(float) * n_local);
    float *Ay_l = (float *)malloc(sizeof(float) * n_local);
    float *Az_l = (float *)malloc(sizeof(float) * n_local);

    if (!Ex_l || !Ey_l || !Ez_l || !Ax_l || !Ay_l || !Az_l)
    {
        if (rank == 0)
            fprintf(stderr, "malloc(local component arrays) failed\n");
        free(Ex_l);
        free(Ey_l);
        free(Ez_l);
        free(Ax_l);
        free(Ay_l);
        free(Az_l);
        free(t_all);
        free(z_all);
        free(z_counts);
        free(z_displs);
        MPI_Finalize();
        return 5;
    }

    /* Progress update cadence (in z-rows per rank) */
    const int progress_every_rows = 10; /* tune: 1, 5, 10, 20... */
    double t_progress0 = 0.0;
    if (rank == 0)
        t_progress0 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Compute local slab */
    int local_done = 0;

    for (int izl = 0; izl < nz_local; ++izl)
    {
        const int izg = iz0_local + izl;
        const double z = z_all[izg];

        const double r_um[3] = {x0_um, y0_um, z};

        for (int it = 0; it < nt; ++it)
        {
            const double tt = t_all[it];

            double E[3], A[3];
            LaserPulse_E(pulse, tt, r_um, E);
            LaserPulse_A(pulse, tt, r_um, A);

            const size_t idx = (size_t)izl * nt + it;
            Ex_l[idx] = (float)E[0];
            Ey_l[idx] = (float)E[1];
            Ez_l[idx] = (float)E[2];
            Ax_l[idx] = (float)A[0];
            Ay_l[idx] = (float)A[1];
            Az_l[idx] = (float)A[2];
        }

        /* progress update: AFTER finishing one z row */
        if (rank == 0 &&
            progress_every_rows > 0 &&
            (izl % progress_every_rows) == 0)
        {
            const int global_done_est =
                (int)((double)izl / (double)nz_local * (double)nz);

            progress_print_rank0(global_done_est, nz, t_progress0);
        }
    }

    /* final flush */
    if (rank == 0)
        progress_print_rank0(nz, nz, t_progress0);

    /* final progress flush */
    {
        int global_done = 0;
        MPI_Reduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
            progress_print_rank0(global_done, nz, t_progress0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* Root arrays */
    float *Ex = NULL, *Ey = NULL, *Ez = NULL, *Ax = NULL, *Ay = NULL, *Az = NULL;
    if (rank == 0)
    {
        const size_t n_tot = (size_t)nz * (size_t)nt;
        Ex = (float *)malloc(sizeof(float) * n_tot);
        Ey = (float *)malloc(sizeof(float) * n_tot);
        Ez = (float *)malloc(sizeof(float) * n_tot);
        Ax = (float *)malloc(sizeof(float) * n_tot);
        Ay = (float *)malloc(sizeof(float) * n_tot);
        Az = (float *)malloc(sizeof(float) * n_tot);

        if (!Ex || !Ey || !Ez || !Ax || !Ay || !Az)
        {
            fprintf(stderr, "malloc(root arrays) failed\n");
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
    }

    /* Prepare counts/displs in FLOAT elements (each z row has nt floats) */
    int *counts_f = (int *)malloc(sizeof(int) * (size_t)size);
    int *displs_f = (int *)malloc(sizeof(int) * (size_t)size);
    if (!counts_f || !displs_f)
    {
        if (rank == 0)
            fprintf(stderr, "malloc(counts_f/displs_f) failed\n");
        MPI_Abort(MPI_COMM_WORLD, 7);
    }
    for (int r = 0; r < size; ++r)
    {
        counts_f[r] = z_counts[r] * nt;
        displs_f[r] = z_displs[r] * nt;
    }

    /* Gather each component slab */
    MPI_Gatherv(Ex_l, (int)n_local, MPI_FLOAT, Ex, counts_f, displs_f, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Ey_l, (int)n_local, MPI_FLOAT, Ey, counts_f, displs_f, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Ez_l, (int)n_local, MPI_FLOAT, Ez, counts_f, displs_f, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(Ax_l, (int)n_local, MPI_FLOAT, Ax, counts_f, displs_f, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Ay_l, (int)n_local, MPI_FLOAT, Ay, counts_f, displs_f, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Az_l, (int)n_local, MPI_FLOAT, Az, counts_f, displs_f, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if (rank == 0)
    {
        /* AXIS1 = t (fastest), AXIS2 = z (slow) */
        DiagAxis1D axis_t = {.name = "t", .long_name = "t", .units = "fs", .vmin = tmin_fs, .vmax = tmax_fs};
        DiagAxis1D axis_z = {.name = "z", .long_name = "z", .units = "um", .vmin = zmin_um, .vmax = zmax_um};

        char fname[256];

        snprintf(fname, sizeof(fname), "%s_Ex.h5", prefix);
        write_component_2d(fname, "Ex", "GV/m", "E_x", Ex, (size_t)nt, (size_t)nz, &axis_t, &axis_z);

        snprintf(fname, sizeof(fname), "%s_Ey.h5", prefix);
        write_component_2d(fname, "Ey", "GV/m", "E_y", Ey, (size_t)nt, (size_t)nz, &axis_t, &axis_z);

        snprintf(fname, sizeof(fname), "%s_Ez.h5", prefix);
        write_component_2d(fname, "Ez", "GV/m", "E_z", Ez, (size_t)nt, (size_t)nz, &axis_t, &axis_z);

        snprintf(fname, sizeof(fname), "%s_Ax.h5", prefix);
        write_component_2d(fname, "Ax", "a.u.", "A_x", Ax, (size_t)nt, (size_t)nz, &axis_t, &axis_z);

        snprintf(fname, sizeof(fname), "%s_Ay.h5", prefix);
        write_component_2d(fname, "Ay", "a.u.", "A_y", Ay, (size_t)nt, (size_t)nz, &axis_t, &axis_z);

        snprintf(fname, sizeof(fname), "%s_Az.h5", prefix);
        write_component_2d(fname, "Az", "a.u.", "A_z", Az, (size_t)nt, (size_t)nz, &axis_t, &axis_z);

        const double t_compute = t1 - t0;
        const double t_gather = t2 - t1;
        printf("[rank0] zt grid: nz=%d nt=%d ranks=%d  compute=%g s  gather=%g s  total=%g s\n",
               nz, nt, size, t_compute, t_gather, (t2 - t0));
        printf("[rank0] wrote: %s_Ex.h5 ... %s_Az.h5\n", prefix, prefix);

        free(Ex);
        free(Ey);
        free(Ez);
        free(Ax);
        free(Ay);
        free(Az);
    }

    free(counts_f);
    free(displs_f);
    free(Ex_l);
    free(Ey_l);
    free(Ez_l);
    free(Ax_l);
    free(Ay_l);
    free(Az_l);
    free(t_all);
    free(z_all);
    free(z_counts);
    free(z_displs);

    MPI_Finalize();
    return 0;
}
