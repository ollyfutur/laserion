#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "laser.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

#include "field_grid.h" /* fg_run_1d/fg_run_2d */

/* Small helper: make output folder only once */
static void ensure_outdir(int rank)
{
    if (rank == 0)
        (void)system("mkdir -p diag_out");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ensure_outdir(rank);

    /* ------------------------- Build a simple pulse ------------------------- */

    /* Physical-ish parameters (units match your codebase: um, fs, GV/m) */
    const double E0 = 150.0;       /* GV/m */
    const double lambda_um = 10.0; /* um */
    const double tau_fs = 200.0;   /* fs (Gaussian RMS) */
    const double phase0 = 0.0;
    const int use_retarded = 1;

    /* Propagation along +z */
    const double k_vec[3] = {0.02, 0.02, 1.0};

    /* Start position for pulse reference (z shift) */
    const double r_start[3] = {0.0, 0.0, 50.0};

    /* Temporal / transverse */
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, tau_fs);

    PlaneWaveProfile trans;
    PlaneWaveProfile_init(&trans);

    /* Linear polarization, angle=0 => one transverse basis direction */
    Polarization pol;
    if (CircularPolarization_init(&pol, k_vec, "right", 0.0) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "LinearPolarization_init failed.\n");
        MPI_Finalize();
        return 1;
    }

    /* SinglePulse */
    SinglePulse pulse;
    if (SinglePulse_init(&pulse,
                         E0, lambda_um,
                         (const TemporalProfile *)&temporal,
                         (const TransverseProfile *)&trans,
                         &pol,
                         phase0,
                         k_vec,
                         use_retarded,
                         r_start) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "SinglePulse_init failed.\n");
        MPI_Finalize();
        return 2;
    }

    /* IMPORTANT: enable A integration so Ax/Ay/Az can be computed */
    /* Choose wide enough bounds and reasonable dt for a quick test */
    SinglePulse_enable_A(&pulse, -2000.0, 2000.0, 0.5);

    /* ------------------------- Options (common) ------------------------- */

    FG_Options opt = fg_default_options();
    opt.root_rank = 0;
    opt.write_mode = FG_WRITE_SPLIT_FILES; /* one file per dataset */

    /* ------------------------- Example 1: Ex(t) 1D ------------------------- */
    if (rank == 0)
        printf("[case 1] Ex(t) at (x,y,z)=(0,0,0)\n");

    {
        FG_Request1D r = {0};
        r.a1 = (FG_AxisSpec){
            .id = FG_T, .kind = FG_AXIS_LINSPACE, .name = "t", .units = "fs", .min = -400.0, .max = 400.0, .n = 1601};
        r.t0_fs = 0.0;
        r.x0_um = 0.0;
        r.y0_um = 0.0;
        r.z0_um = 0.0;

        opt.quantity_mask = FG_Q_E;
        opt.component_mask = FG_C_X;

        int rc = fg_run_1d((const LaserPulse *)&pulse, &r, &opt, "diag_out/ex_vs_t", MPI_COMM_WORLD);
        if (rc != 0 && rank == 0)
            fprintf(stderr, "case 1 failed: rc=%d\n", rc);
    }

    /* ------------------------- Example 2: Ay(z) 1D ------------------------- */
    if (rank == 0)
        printf("[case 2] Ay(z) at (t,x,y)=(0,0,0)\n");

    {
        FG_Request1D r = {0};
        r.a1 = (FG_AxisSpec){
            .id = FG_Z, .kind = FG_AXIS_LINSPACE, .name = "z", .units = "um", .min = -200.0, .max = 200.0, .n = 1601};
        r.t0_fs = 0.0;
        r.x0_um = 0.0;
        r.y0_um = 0.0;
        r.z0_um = 0.0;

        opt.quantity_mask = FG_Q_A;
        opt.component_mask = FG_C_Y;

        int rc = fg_run_1d((const LaserPulse *)&pulse, &r, &opt, "diag_out/ay_vs_z", MPI_COMM_WORLD);
        if (rc != 0 && rank == 0)
            fprintf(stderr, "case 2 failed: rc=%d\n", rc);
    }

    /* ------------------------- Example 3: Ey(t,z) 2D ------------------------ */
    if (rank == 0)
        printf("[case 3] Ey(t,z) at (x,y)=(0,0)\n");

    {
        FG_Request2D r = {0};
        r.a1 = (FG_AxisSpec){
            .id = FG_T, .kind = FG_AXIS_LINSPACE, .name = "t", .units = "fs", .min = -400.0, .max = 400.0, .n = 801};
        r.a2 = (FG_AxisSpec){
            .id = FG_Z, .kind = FG_AXIS_LINSPACE, .name = "z", .units = "um", .min = -300.0, .max = 300.0, .n = 601};
        r.t0_fs = 0.0;
        r.x0_um = 0.0;
        r.y0_um = 0.0;
        r.z0_um = 0.0;

        opt.quantity_mask = FG_Q_E;
        opt.component_mask = FG_C_Y;

        int rc = fg_run_2d((const LaserPulse *)&pulse, &r, &opt, "diag_out/ey_tz", MPI_COMM_WORLD);
        if (rc != 0 && rank == 0)
            fprintf(stderr, "case 3 failed: rc=%d\n", rc);
    }

    /* ------------------------- Example 4: Ex(x,y) 2D ------------------------ */
    if (rank == 0)
        printf("[case 4] Ex(x,y) at (t,z)=(0,0)\n");

    {
        FG_Request2D r = {0};
        r.a1 = (FG_AxisSpec){
            .id = FG_X, .kind = FG_AXIS_LINSPACE, .name = "x", .units = "um", .min = -50.0, .max = 50.0, .n = 801};
        r.a2 = (FG_AxisSpec){
            .id = FG_Y, .kind = FG_AXIS_LINSPACE, .name = "y", .units = "um", .min = -50.0, .max = 50.0, .n = 801};
        r.t0_fs = 0.0;
        r.z0_um = 0.0;
        r.x0_um = 0.0;
        r.y0_um = 0.0;

        opt.quantity_mask = FG_Q_E;
        opt.component_mask = FG_C_X;

        int rc = fg_run_2d((const LaserPulse *)&pulse, &r, &opt, "diag_out/ex_xy", MPI_COMM_WORLD);
        if (rc != 0 && rank == 0)
            fprintf(stderr, "case 4 failed: rc=%d\n", rc);
    }

    /* ------------------------- Example 5: multiple datasets ---------------- */
    if (rank == 0)
        printf("[case 5] Multi-dataset (Ex,Ez,Ax,Az) on (t,x) grid\n");

    {
        FG_Request2D r = {0};
        r.a1 = (FG_AxisSpec){
            .id = FG_T, .kind = FG_AXIS_LINSPACE, .name = "t", .units = "fs", .min = -300.0, .max = 300.0, .n = 1201};
        r.a2 = (FG_AxisSpec){
            .id = FG_X, .kind = FG_AXIS_LINSPACE, .name = "x", .units = "um", .min = -40.0, .max = 40.0, .n = 801};
        r.t0_fs = 0.0;
        r.y0_um = 0.0;
        r.z0_um = 0.0;
        r.x0_um = 0.0;

        opt.quantity_mask = (FG_Q_E | FG_Q_A);
        opt.component_mask = (FG_C_X | FG_C_Z); /* Ex, Ez, Ax, Az */

        int rc = fg_run_2d((const LaserPulse *)&pulse, &r, &opt, "diag_out/multi_tx", MPI_COMM_WORLD);
        if (rc != 0 && rank == 0)
            fprintf(stderr, "case 5 failed: rc=%d\n", rc);
    }

    if (rank == 0)
        printf("Done. Wrote diagnostics under ./diag_out (split files).\n");

    MPI_Finalize();
    return 0;
}
