#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "base.h"               /* for LaserPulse_destroy */
#include "laser.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

#include "mdf_grid.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("test_mdf: running with %d MPI ranks\n", size);
        fflush(stdout);
    }

    /* ------------------------------------------------------------
       Allocate profiles with sufficient lifetime (scope = main)
       ------------------------------------------------------------ */

    /* Temporal: Gaussian, tau = 1000 fs */
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, 100.0);

    /* Transverse: Gaussian (HG00), w0 = 200 um, focus zf = 0 */
    GaussianTransverse transverse;
    GaussianTransverse_init(&transverse, 30.0, 0.0);

    /* Polarization: circular, k along +z */
    Polarization pol;
    const double k_vec[3] = {0.0, 0.0, 1.0};
    if (CircularPolarization_init(&pol, k_vec, "right", 0.0) != 0) {
        if (rank == 0) fprintf(stderr, "CircularPolarization_init failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ------------------------------------------------------------
       Build pulse (SinglePulse)
       ------------------------------------------------------------ */

    SinglePulse sp;
    {
        const double E0_GVm = 200.0;
        const double lambda_um = 1.0;
        const double phase0 = 0.0;
        const int use_retarded_time = 1;
        const double r_start[3] = {0.0, 0.0, 0.0};

        int rc = SinglePulse_init(&sp,
                                  E0_GVm,
                                  lambda_um,
                                  &temporal.base,
                                  &transverse.base,
                                  &pol,
                                  phase0,
                                  k_vec,
                                  use_retarded_time,
                                  r_start);
        if (rc != 0) {
            if (rank == 0) fprintf(stderr, "SinglePulse_init failed (rc=%d)\n", rc);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    const LaserPulse *pulse = &sp.base;

    /* ------------------------------------------------------------
       MDF options: Helium, region x=y=z in [-10,10] um
       ------------------------------------------------------------ */

    const int Z_list[] = {1, 2};

    MDFGridOptions opt;
    opt.species = "He";
    opt.Z_list  = Z_list;
    opt.nZ      = 2;

    opt.ion_model = NULL;

    /* MDF_build requires explicit t window; tau=1000 fs -> choose ±6000 fs */
    opt.tmin_fs = -600.0;
    opt.tmax_fs =  600.0;
    opt.dt_fs   = 0.0;

    opt.envelope_cut = 0.0;

    opt.xmin = -10.0; opt.xmax =  10.0;
    opt.ymin = -10.0; opt.ymax =  10.0;
    opt.zmin = -10.0; opt.zmax =  10.0;

    /* sampling points in real space (distributed over MPI) */
    opt.nx = 32;
    opt.ny = 32;
    opt.nz = 32;

    /* MDF projection */
    opt.kind = MDF_GRID_F_PX_PY;

    /* bins in m_e c */
    opt.nbins1 = 300; opt.p1min = -0.15; opt.p1max = 0.15;
    opt.nbins2 = 300; opt.p2min = -0.15; opt.p2max = 0.15;

    opt.normalize_sum_to_1 = 1;

    opt.dataset_name = "f";
    opt.label        = "f(p_x,p_y)";
    opt.units        = "1";
    opt.file_suffix  = "_mdf_pxpy.h5";
    opt.root_rank    = 0;

    const char *prefix = "out/he_circ_10um_w200_tau1000_E200";

    int mrc = mdfgrid_run(pulse, &opt, prefix, MPI_COMM_WORLD);

    if (rank == 0) {
        if (mrc == 0) {
            printf("MDF OK: wrote %s%s\n", prefix, opt.file_suffix);
        } else {
            printf("MDF FAILED: rc=%d\n", mrc);
        }
        fflush(stdout);
    }

    /* ------------------------------------------------------------
       Destroy pulse (generic vtable-based destructor)
       ------------------------------------------------------------ */
    LaserPulse_destroy(&sp.base);

    MPI_Finalize();
    return (mrc == 0) ? 0 : 1;
}

