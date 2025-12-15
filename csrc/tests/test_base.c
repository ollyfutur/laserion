#include <stdio.h>
#include <math.h>

#include "base.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"
#include "ionization_model.h"
#include "core.h"

static int assert_near(const char *name, double a, double b, double tol)
{
    double err = fabs(a - b);
    if (err > tol) {
        printf("ASSERT FAILED: %s: a=%.17g b=%.17g |a-b|=%.3e tol=%.3e\n", name, a, b, err, tol);
        return 1;
    }
    return 0;
}

static int assert_true(const char *name, int cond)
{
    if (!cond) {
        printf("ASSERT FAILED: %s\n", name);
        return 1;
    }
    return 0;
}

int main(void)
{
    int fail = 0;

    /* ===================== quick ADK sanity (kept) ===================== */
    {
        ADKModel adk;
        ADKModel_init(&adk);
        double w0 = IonizationModel_rate((IonizationModel *)&adk, 0.0, "H", 1);
        fail |= assert_true("adk_rate_zero_is_zero", w0 == 0.0);
    }

    /* ===================== SinglePulse tests =========================== */

    /* Components */
    GaussianTemporal gt;
    GaussianTemporal_init(&gt, 100.0);

    PlaneWaveProfile pw;
    PlaneWaveProfile_init(&pw);

    /* Linear polarization, k || z, angle=0 => x-like at phase=0 */
    Polarization pol;
    double kvec[3] = {0.0, 0.0, 1.0};
    fail |= assert_true("linear_pol_init_ok", LinearPolarization_init(&pol, kvec, 0.0) == 0);

    /* Build pulse */
    SinglePulse sp;
    double rstart[3] = {0.0, 0.0, 0.0};
    double E0 = 150.0;
    double lambda = 10.0; /* µm */
    fail |= assert_true("singlepulse_init_ok",
        SinglePulse_init(&sp, E0, lambda,
                         (TemporalProfile *)&gt,
                         (TransverseProfile *)&pw,
                         &pol,
                         0.0, /* phase0 */
                         kvec,
                         1,   /* use_retarded_time */
                         rstart) == 0);

    /* Test 1: at origin t=0 => env=1, phase=0 => E=[E0,0,0] */
    {
        double r[3] = {0.0, 0.0, 0.0};
        double E[3];
        LaserPulse_E((LaserPulse *)&sp, 0.0, r, E);

        printf("SinglePulse E(t=0,r=0) = [%.17g %.17g %.17g]\n", E[0], E[1], E[2]);

        fail |= assert_near("sp_E0_x", E[0], E0, 1e-12);
        fail |= assert_near("sp_E0_y", E[1], 0.0, 1e-12);
        fail |= assert_near("sp_E0_z", E[2], 0.0, 1e-12);
    }

    /* Test 2: retarded time: choose z and t=z/c => t_eff=0 => same as origin */
    {
        double c = 0.299792458;
        double z = 2.0; /* µm */
        double t = z / c; /* fs */
        double r[3] = {0.0, 0.0, z};

        double E[3];
        LaserPulse_E((LaserPulse *)&sp, t, r, E);

        printf("SinglePulse retarded-time test at z=2um, t=z/c: E=[%.17g %.17g %.17g]\n",
               E[0], E[1], E[2]);

        fail |= assert_near("sp_retarded_E_x", E[0], E0, 1e-10);
        fail |= assert_near("sp_retarded_E_y", E[1], 0.0, 1e-10);
        fail |= assert_near("sp_retarded_E_z", E[2], 0.0, 1e-10);
    }

    /* Test 3: phase0 = pi/2 => at t_eff=0, cos(phase)=0 -> near-zero field */
    {
        SinglePulse sp2;
        fail |= assert_true("singlepulse_init_phase_pi2_ok",
            SinglePulse_init(&sp2, E0, lambda,
                             (TemporalProfile *)&gt,
                             (TransverseProfile *)&pw,
                             &pol,
                             3.14159265358979323846 / 2.0,
                             kvec,
                             1,
                             rstart) == 0);

        double r[3] = {0.0, 0.0, 0.0};
        double E[3];
        LaserPulse_E((LaserPulse *)&sp2, 0.0, r, E);

        printf("SinglePulse phase0=pi/2 at origin: E=[%.17g %.17g %.17g]\n", E[0], E[1], E[2]);

        fail |= assert_true("sp_phase_pi2_small", fabs(E[0]) < 1e-9);
    }

    /* Test 4: MultiPulse sum of two identical pulses => doubles */
    {
        const LaserPulse *arr[2];
        arr[0] = (const LaserPulse *)&sp;
        arr[1] = (const LaserPulse *)&sp;

        MultiPulse mp;
        fail |= assert_true("multipulse_init_ok", MultiPulse_init(&mp, arr, 2) == 0);

        double r[3] = {0.0, 0.0, 0.0};
        double E[3];
        LaserPulse_E((LaserPulse *)&mp, 0.0, r, E);

        printf("MultiPulse (2x) at origin: E=[%.17g %.17g %.17g]\n", E[0], E[1], E[2]);

        fail |= assert_near("mp_double", E[0], 2.0*E0, 1e-12);
    }

    if (fail) {
        printf("\nTESTS FAILED.\n");
        return 1;
    }

    printf("\nAll tests passed.\n");
    return 0;
}

