#ifndef TEMPORAL_PROFILE_H
#define TEMPORAL_PROFILE_H

#include "base.h"

/*
 * Concrete temporal profiles.
 *
 * Each profile embeds TemporalProfile as its first member and provides
 * an init() function.
 */

/* ===================== Gaussian temporal ===================== */

typedef struct
{
    TemporalProfile base; /* must be first */
    double tau_fs;
} GaussianTemporal;

void GaussianTemporal_init(GaussianTemporal *g, double tau_fs);

/* ===================== (future) Flat-top ===================== */
/*
typedef struct {
    TemporalProfile base;
    double t_rise, t_flat, t_fall;
} FlatTopTemporal;

void FlatTopTemporal_init(FlatTopTemporal *f, ...);
*/

#endif /* TEMPORAL_PROFILE_H */
