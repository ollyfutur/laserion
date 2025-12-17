// tests/test_field_h5_single.c
//
// SinglePulse + simple linear polarization
// Outputs several field diagnostics in OSIRIS-minimal HDF5 format.
//
// Requires:
//   - core.h (LaserPulse_E / LaserPulse_A, SinglePulse_init, etc.)
//   - temporal_profile.h (GaussianTemporal_init)
//   - transverse_profile.h (PlaneWaveProfile_init)
//   - polarization.h (LinearPolarization_init)
//   - diag_h5.h (diag_h5_write_field_1d/2d)
//
// Build/link needs HDF5.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

#include "core.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"
#include "diag_h5.h"

static void mkdir_p(const char *path)
{
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

static float *xmalloc_f(size_t n)
{
    float *p = (float*)malloc(n * sizeof(float));
    if (!p) {
        fprintf(stderr, "Allocation failed (n=%zu)\n", n);
        exit(2);
    }
    return p;
}

int main(void)
{
    mkdir_p("out");

    // ----------------- Define a simple SinglePulse -----------------
    // Laser parameters (pick something reasonable for a test)
    const double E0_GVm    = 150.0;   // peak field [GV/m]
    const double lambda_um = 10.0;    // wavelength [um]
    const double cep_rad   = 0.0;     // CEP [rad]

    // Propagation direction (not necessarily normalized in your code; keep consistent with your usage)
    double k[3]  = {0.0, 0.0, 1.0};

    // Pulse center position (r0): one position only
    double r0[3] = {0.0, 0.0, 0.0};

    // Temporal profile: Gaussian with RMS duration tau [fs]
    GaussianTemporal tp;
    GaussianTemporal_init(&tp, 800.0);

    // Transverse: plane wave (no focusing)
    PlaneWaveProfile tr;
    PlaneWaveProfile_init(&tr);

    // Polarization: linear at +0 degrees in the transverse plane
    Polarization pol;
    if (LinearPolarization_init(&pol, k, 90.0) != 0) {
        fprintf(stderr, "LinearPolarization_init failed\n");
        return 1;
    }

    // Build pulse
    SinglePulse p;
    if (SinglePulse_init(&p,
                         E0_GVm,
                         lambda_um,
                         (const TemporalProfile*)&tp,
                         (const TransverseProfile*)&tr,
                         &pol,
                         cep_rad,
                         k,
                         1,
                         r0) != 0)
    {
        fprintf(stderr, "SinglePulse_init failed\n");
        return 1;
    }

    // Convenience pointer for polymorphic calls
    const LaserPulse *lp = (const LaserPulse*)&p;

    // ----------------- Diagnostic 1: Ey(t) at fixed (x,y,z) -----------------
    {
        const double x_um = 0.0, y_um = 3.0, z_um = 0.0;
        const double tmin_fs = -2000.0, tmax_fs = 2000.0;
        const size_t nt = 2001;

        float *ey = xmalloc_f(nt);

        for (size_t i = 0; i < nt; ++i) {
            double t_fs = tmin_fs + (tmax_fs - tmin_fs) * (double)i / (double)(nt - 1);
            double r_um[3] = {x_um, y_um, z_um};
            double E[3];
            LaserPulse_E(lp, t_fs, r_um, E);
            ey[i] = (float)E[1]; // Ey
        }

        DiagAxis1D ax_t = {
            .name = "t", .long_name = "t",
            .units = "fs",
            .vmin = tmin_fs, .vmax = tmax_fs
        };

        // Per your rules: time series -> TIME=0, ITER=0
        if (diag_h5_write_field_1d("out/ey_timeseries_t.h5",
                                   "E_y",
                                   "GV/m",
                                   0.0,
                                   0,
                                   ey, nt,
                                   &ax_t) != 0)
        {
            fprintf(stderr, "Failed writing out/ey_timeseries_t.h5\n");
            free(ey);
            return 1;
        }

        free(ey);
    }

    // ----------------- Diagnostic 2: Ay(t) at fixed (x,y,z) -----------------
    // If SinglePulse A is not available in your core, comment this block out.
        SinglePulse_enable_A(&p, -3000.0, 3000.0, 0.05);    {
        const double x_um = 0.0, y_um = 3.0, z_um = 0.0;
        const double tmin_fs = -2000.0, tmax_fs = 2000.0;
        const size_t nt = 4001;

        float *ay = xmalloc_f(nt);

        for (size_t i = 0; i < nt; ++i) {
            double t_fs = tmin_fs + (tmax_fs - tmin_fs) * (double)i / (double)(nt - 1);
            double r_um[3] = {x_um, y_um, z_um};
            double A[3];
            LaserPulse_A(lp, t_fs, r_um, A);
            ay[i] = (float)A[1]; // Ay
        }

        DiagAxis1D ax_t = {
            .name = "t", .long_name = "t",
            .units = "fs",
            .vmin = tmin_fs, .vmax = tmax_fs
        };

        // TIME=0, ITER=0 for a time series
        if (diag_h5_write_field_1d("out/ay_timeseries_t.h5",
                                   "ay",
                                   "GV/m fs",
                                   0.0,
                                   0,
                                   ay, nt,
                                   &ax_t) != 0)
        {
            fprintf(stderr, "Failed writing out/ay_timeseries_t.h5\n");
            free(ay);
            return 1;
        }

        free(ay);
    }

    // ----------------- Diagnostic 3: Ey(z) lineout at fixed (x,y,t) -----------------
    {
        const double x_um = -4.0, y_um = 3.0;
        const double t_fs = 0.0;
        const double zmin_um = -200.0, zmax_um = 200.0;
        const size_t nz = 801;

        float *ey = xmalloc_f(nz);

        for (size_t i = 0; i < nz; ++i) {
            double z_um = zmin_um + (zmax_um - zmin_um) * (double)i / (double)(nz - 1);
            double r_um[3] = {x_um, y_um, z_um};
            double E[3];
            LaserPulse_E(lp, t_fs, r_um, E);
            ey[i] = (float)E[1];
        }

        DiagAxis1D ax_z = {
            .name = "z", .long_name = "z",
            .units = "\\mu m",
            .vmin = zmin_um, .vmax = zmax_um
        };

        // Fixed-time plot -> TIME=t, ITER=int(t*10000)
        int iter = (int)llround(t_fs * 10000.0);

        if (diag_h5_write_field_1d("out/ey_lineout_z.h5",
                                   "E_y",
                                   "GV/m",
                                   t_fs,
                                   iter,
                                   ey, nz,
                                   &ax_z) != 0)
        {
            fprintf(stderr, "Failed writing out/ey_lineout_z.h5\n");
            free(ey);
            return 1;
        }

        free(ey);
    }

    // ----------------- Diagnostic 4: Ey(y,z) slice at fixed x and t -----------------
    // We store data as [nz, ny] with y as AXIS1 (fast) and z as AXIS2 (slow).
    {
        const double x_um = -4.0;
        const double t_fs = 1000.0;

        const double ymin_um = -20.0, ymax_um = 20.0;
        const double zmin_um = -200.0, zmax_um = 200.0;

        const size_t ny = 241;
        const size_t nz = 401;

        float *ey = xmalloc_f(ny * nz);

        for (size_t iz = 0; iz < nz; ++iz) {
            double z_um = zmin_um + (zmax_um - zmin_um) * (double)iz / (double)(nz - 1);
            for (size_t iy = 0; iy < ny; ++iy) {
                double y_um = ymin_um + (ymax_um - ymin_um) * (double)iy / (double)(ny - 1);
                double r_um[3] = {x_um, y_um, z_um};
                double E[3];
                LaserPulse_E(lp, t_fs, r_um, E);
                ey[iz * ny + iy] = (float)E[1];
            }
        }

        DiagAxis1D ax_y = {
            .name = "y", .long_name = "y",
            .units = "\\mu m",
            .vmin = ymin_um, .vmax = ymax_um
        };
        DiagAxis1D ax_z = {
            .name = "z", .long_name = "z",
            .units = "\\mu m",
            .vmin = zmin_um, .vmax = zmax_um
        };

        int iter = (int)llround(t_fs * 10000.0);

        if (diag_h5_write_field_2d("out/ey_slice_yz.h5",
                                   "E_y",
                                   "GV/m",
                                   t_fs,
                                   iter,
                                   ey,
                                   ny, nz,
                                   &ax_y, &ax_z) != 0)
        {
            fprintf(stderr, "Failed writing out/ey_slice_yz.h5\n");
            free(ey);
            return 1;
        }

        free(ey);
    }

    printf("Wrote HDF5 diagnostics under ./out/\n");
    return 0;
}

