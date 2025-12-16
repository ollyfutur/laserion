#include <math.h>
#include "temporal_profile.h"

/* --------------------- Gaussian temporal --------------------- */

static double gaussian_temporal_eval(const TemporalProfile *base, double t_fs)
{
    const GaussianTemporal *self = (const GaussianTemporal *)base;
    const double tau = self->tau_fs;
    return exp(-2.0 * (t_fs * t_fs) / (tau * tau));
}

static void gaussian_temporal_destroy(TemporalProfile *base)
{
    (void)base;
}

static const TemporalProfileVTable GAUSSIAN_TEMPORAL_VTABLE = {
    .eval = gaussian_temporal_eval,
    .destroy = gaussian_temporal_destroy};

void GaussianTemporal_init(GaussianTemporal *g, double tau_fs)
{
    g->base.vt = &GAUSSIAN_TEMPORAL_VTABLE;
    g->tau_fs = tau_fs;
}
