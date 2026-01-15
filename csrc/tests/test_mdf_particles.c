#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "laser.h"
#include "mdf_particles.h"
#include "diag_h5.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ---------------- Laser construction ---------------- */

    GaussianTemporal gt;
    GaussianTemporal_init(&gt, 1000.0);

    GaussianTransverse gtr;
    GaussianTransverse_init(&gtr, 200.0, 0.0);

    Polarization pol;
    const double k_vec[3] = {0.0, 0.0, 1.0};
    if (CircularPolarization_init(&pol, k_vec, "right", 0.0) != 0)
    {
        if (rank == 0) fprintf(stderr, "LinearPolarization_init failed\n");
        MPI_Finalize();
        return 1;
    }

    SinglePulse sp;
    const double r_start[3] = {0.0, 0.0, 0.0};

    if (SinglePulse_init(&sp,
                         200.0,      /* E0 */
                         10,      /* wavelength_um */
                         &gt.base,
                         &gtr.base,
                         &pol,
                         0.0,      /* phase0 */
                         k_vec,
                         1,        /* use_retarded_time */
                         r_start) != 0)
    {
        if (rank == 0) fprintf(stderr, "SinglePulse_init failed\n");
        MPI_Finalize();
        return 2;
    }

    const LaserPulse *pulse = &sp.base;

    /* ---------------- MDF -> particles options ---------------- */

    static const int Z_list[] = {1};

    MDFParticlesOptions opt;
    memset(&opt, 0, sizeof(opt));

    opt.species      = "He";
    opt.Z_list       = Z_list;
    opt.nZ           = 1;
    opt.ion_model    = NULL;   /* MDF_build will create ADK internally */
    opt.envelope_cut = 1e-6;

    opt.tmin_fs = -2000.0;
    opt.tmax_fs =  2000.0;
    opt.dt_fs   =  0.05;

    opt.xmin = -2.0;  opt.xmax = 2.0;
    opt.ymin = -2.0;  opt.ymax = 2.0;
    opt.zmin = -2.0;  opt.zmax = 2.0;

    opt.nx = 4;
    opt.ny = 4;
    opt.nz = 4;

    opt.ppc = 1000;

    opt.mdf_at_particle_position = 0;

    opt.dataset_name = "electrons";
    opt.file_suffix  = "_particles.h5";

    /* This matches your diag_h5.h */
    opt.particle_map = DIAG_PARTICLE_MAP_ZXY_PZPXPY;

    opt.root_rank = 0;
    opt.seed      = 0;

    const int rc = mdf_particles_run(pulse, &opt, "bin/mdf_particles", MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (rc == 0)
            printf("Wrote particle diagnostic: bin/mdf_particles_particles.h5\n");
        else
            fprintf(stderr, "mdf_particles_run failed (rc=%d)\n", rc);
    }

    MPI_Finalize();
    return rc;
}

