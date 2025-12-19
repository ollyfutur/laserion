#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "laser.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

#include "diag_grid.h"
#include "ionization_model.h"
#include "ionization_grid.h"

static double deg2rad(double d) { return d * (3.14159265358979323846 / 180.0); }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* --------------------- laser definition (tilted) --------------------- */

    /* Representative parameters (adjust to your needs) */
    const double E0_GVm = 150.0;        /* peak E amplitude (units consistent with your ADK usage) */
    const double lambda_um = 10.0;      /* wavelength [um] */
    const double tau_fs = 200.0;        /* Gaussian RMS duration [fs] */
    const double w0_um = 50.0;           /* waist [um] */
    const double zf_um = 0.0;           /* focus position along beam z' [um] */
    const double phase0 = 0.0;

    /* Tilt: ky=0, choose angle in x–z plane */
    const double tilt_deg = 15.0;
    const double th = deg2rad(tilt_deg);

    /* k_vec can be any nonzero vector; SinglePulse_init uses its direction */
    const double k_vec[3] = { sin(th), 0.0, cos(th) };

    /* Pulse "start" reference position r_start [um] */
    const double r_start[3] = { 0.0, 0.0, 0.0 };

    /* Temporal / transverse profiles */
    GaussianTemporal temporal;
    GaussianTemporal_init(&temporal, tau_fs);

    GaussianTransverse transverse;
    GaussianTransverse_init(&transverse, w0_um, zf_um);

    /* Linear polarization: angle_deg rotates transverse basis around k_hat.
       angle_deg=0 means “default e1 direction” from your basis constructor. */
    Polarization pol;
    if (LinearPolarization_init(&pol, k_vec, 0.0) != 0)
    {
        if (rank == 0) fprintf(stderr, "LinearPolarization_init failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* Construct pulse */
    SinglePulse sp;
    if (SinglePulse_init(&sp,
                         E0_GVm,
                         lambda_um,
                         (const TemporalProfile *)&temporal,
                         (const TransverseProfile *)&transverse,
                         &pol,
                         phase0,
                         k_vec,
                         /* use_retarded_time */ 1,
                         r_start) != 0)
    {
        if (rank == 0) fprintf(stderr, "SinglePulse_init failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    /* Optional: A integration not needed for ionization, so do not enable A. */

    /* ----------------------- ionization diagnostic ------------------------ */

    /* Choose species and charge states to record */
    const char *species = "He";
    const int Z_list[] = {1, 2};
    const size_t nZ = sizeof(Z_list) / sizeof(Z_list[0]);

    ADKModel adk;
    ADKModel_init(&adk);

    IonGridOptions ion = (IonGridOptions){0};
    ion.species = species;
    ion.Z_list = Z_list;
    ion.nZ = nZ;
    ion.ion_model = (const IonizationModel *)&adk;

    /* Time integration window for probability:
       Use a window that covers the pulse passage. */
    ion.tmin_fs = -6.0 * tau_fs;
    ion.tmax_fs = +6.0 * tau_fs;
    ion.dt_fs   = 0.5;               /* adjust accuracy vs cost */
    ion.write_total = 1;

    ion.run = dg_default_run_options();
    ion.run.root_rank = 0;
    ion.run.write_time_iter_0 = 1;   /* keep your convention TIME=0, ITER=0 */

    /* 2D grid request: x–z slice (y fixed, t fixed) */
    DG_Request2D req = (DG_Request2D){0};

    req.fixed.t_fs = 0.0;   /* “slice time” attribute only; probability uses ion.tmin..tmax */
    req.fixed.y_um = 0.0;
    req.fixed.x_um = 0.0;   /* will be overwritten by axis */
    req.fixed.z_um = 0.0;   /* will be overwritten by axis */

    req.a1.id = DG_AXIS_X;
    req.a1.kind = DG_AXIS_LINSPACE;
    req.a1.name = "x";
    req.a1.units = "um";
    req.a1.min = -40.0;
    req.a1.max = +40.0;
    req.a1.n = 401;         /* odd count includes 0 */

    req.a2.id = DG_AXIS_Z;
    req.a2.kind = DG_AXIS_LINSPACE;
    req.a2.name = "z";
    req.a2.units = "um";
    req.a2.min = -200.0;
    req.a2.max = +200.0;
    req.a2.n = 801;

    /* Output prefix (split files): ensure directory exists before running. */
    const char *prefix = "out/ion_tilt";

    const int rc = iongrid_run_2d((const LaserPulse *)&sp, &req, &ion, prefix, MPI_COMM_WORLD);
    if (rank == 0)
    {
        if (rc != 0) fprintf(stderr, "iongrid_run_2d failed with code %d\n", rc);
        else fprintf(stdout, "Wrote ionization H5 files with prefix: %s\n", prefix);
        fprintf(stdout, "MPI ranks used: %d\n", size);
    }

    MPI_Finalize();
    return (rc == 0) ? 0 : 1;
}

