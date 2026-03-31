#include "laser_build.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
    if (!p)
    {
        fprintf(stderr, "fatal: out of memory\n");
        exit(1);
    }
    return p;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "fatal: out of memory\n");
        exit(1);
    }
    return p;
}

static void xfree(void *p) { free(p); }

static double deg2rad(double deg) { return deg * (M_PI / 180.0); }

static int build_temporal(const InputLaserSpec *in, const TemporalProfile **out_tp)
{
    if (!in || !out_tp)
        return 1;

    switch (in->temporal_type)
    {
    case TEMP_GAUSSIAN:
    {
        GaussianTemporal *gt = (GaussianTemporal *)xmalloc(sizeof(*gt));
        GaussianTemporal_init(gt, in->tau);
        *out_tp = (const TemporalProfile *)gt;
        return 0;
    }
    default:
        return 2;
    }
}

static int build_transverse(const InputLaserSpec *in, const TransverseProfile **out_tr)
{
    if (!in || !out_tr)
        return 1;

    switch (in->transverse_type)
    {
    case TRANS_PLANE:
    {
        PlaneWaveProfile *p = xmalloc(sizeof(*p));
        PlaneWaveProfile_init(p);
        *out_tr = (const TransverseProfile *)p;
        return 0;
    }
    case TRANS_GAUSSIAN:
    {
        GaussianTransverse *g = (GaussianTransverse *)xmalloc(sizeof(*g));
        GaussianTransverse_init(g, in->w0, in->zf);
        *out_tr = (const TransverseProfile *)g;
        return 0;
    }
    case TRANS_HERMITE:
    {
        HermiteTransverse *h = (HermiteTransverse *)xmalloc(sizeof(*h));
        HermiteTransverse_init(h, in->w0, in->zf, in->herm_l, in->herm_m);
        *out_tr = (const TransverseProfile *)h;
        return 0;
    }
    default:
        return 2;
    }
}

static int build_polarization(const InputLaserSpec *in, Polarization *out_pol)
{
    if (!in || !out_pol)
        return 1;

    // Your input uses angle in degrees (good: API expects degrees).
    // Your input uses delta likely in degrees (90.0). polarization.h expects radians.
    double delta_rad = deg2rad(in->delta);

    switch (in->polarization)
    {
    case POL_LINEAR:
        return LinearPolarization_init(out_pol, in->k_vec, in->angle);

    case POL_CIRCULAR:
    {

        const char *s = (in->sense == SENSE_LEFT) ? "left" : "right";
        return CircularPolarization_init(out_pol, in->k_vec, s, in->angle);
    }

    case POL_JONES:
    {
        // Build transverse basis rotated by angle, then set (p1,p2,delta)
        double e1[3], e2[3];
        int rc = Polarization_make_basis_angle(in->k_vec, in->angle, e1, e2);
        if (rc != 0)
            return rc;
        return Polarization_init(out_pol, e1, e2, in->p1, in->p2, delta_rad);
    }

    default:
        return 2;
    }
}

int BuiltLasers_build(const InputSimSpec *sim, BuiltLasers *out)
{
    if (!sim || !out)
        return 1;
    memset(out, 0, sizeof(*out));

    const size_t n = sim->lasers.count;
    if (n == 0)
        return 2;

    out->count = n;
    out->single = (SinglePulse *)xcalloc(n, sizeof(SinglePulse));
    out->temporal = (const TemporalProfile **)xcalloc(n, sizeof(TemporalProfile *));
    out->trans = (const TransverseProfile **)xcalloc(n, sizeof(TransverseProfile *));
    out->pulse_ptrs = (const LaserPulse **)xcalloc(n, sizeof(LaserPulse *));

    for (size_t i = 0; i < n; ++i)
    {
        const InputLaserSpec *in = &sim->lasers.items[i];

        // 1) sub-objects
        int rc = build_temporal(in, &out->temporal[i]);
        if (rc != 0)
            return 10 + rc;

        rc = build_transverse(in, &out->trans[i]);
        if (rc != 0)
            return 20 + rc;

        Polarization pol;
        rc = build_polarization(in, &pol);
        if (rc != 0)
            return 30 + rc;

        // 2) pulse
        rc = SinglePulse_init(&out->single[i],
                              in->E0,
                              in->wavelength,
                              out->temporal[i],
                              out->trans[i],
                              &pol,
                              deg2rad(in->phase0),
                              in->k_vec,
                              in->use_retarded_time ? 1 : 0,
                              in->r_start);
        if (rc != 0)
            return 40 + rc;

        out->pulse_ptrs[i] = (const LaserPulse *)&out->single[i];

        /* Maxwell correction (optional): longitudinal E field for ∇·E = 0. */
        if (in->maxwell_correction)
            SinglePulse_enable_maxwell_correction(&out->single[i], 0.0 /* auto h */);

        /* Enable numeric A(t,r) integration so field_cache can write Ax/Ay/Az. */
        {
            const double tmin_fs = sim->grid.t_min;
            const double tmax_fs = sim->grid.t_max;

            /* "Slower but safer" option: smaller dt for A integral to reduce artefacts. */
            const double dtA_fs = sim->grid.dt / 1.0; /* change 10.0 -> 1.0 for faster */

            SinglePulse_enable_A(&out->single[i], tmin_fs, tmax_fs, dtA_fs);
        }
    }

    // Build a sum pulse if multiple lasers exist
    if (n > 1)
    {
        int rc = MultiPulse_init(&out->sum, out->pulse_ptrs, n);
        if (rc != 0)
            return 50 + rc;
        out->has_sum = 1;
        /* Enable A for the composite pulse as well (field_cache uses the active pulse). */
        {
            const double tmin_fs = sim->grid.t_min;
            const double tmax_fs = sim->grid.t_max;
            const double dtA_fs = sim->grid.dt / 1.0; /* match above */
            MultiPulse_enable_A(&out->sum, tmin_fs, tmax_fs, dtA_fs);
        }
    }

    return 0;
}

void BuiltLasers_free(BuiltLasers *b)
{
    if (!b)
        return;

    // free temporal/transverse allocations (malloc'ed concrete objects)
    if (b->temporal)
    {
        for (size_t i = 0; i < b->count; ++i)
        {
            // Optional: call interface destroy if you ever set it non-NULL
            // TemporalProfile_destroy((TemporalProfile*)b->temporal[i]);
            xfree((void *)b->temporal[i]);
        }
    }
    if (b->trans)
    {
        for (size_t i = 0; i < b->count; ++i)
        {
            // TransverseProfile_destroy((TransverseProfile*)b->trans[i]);
            xfree((void *)b->trans[i]);
        }
    }

    xfree(b->pulse_ptrs);
    xfree(b->single);
    xfree(b->temporal);
    xfree(b->trans);

    memset(b, 0, sizeof(*b));
}
