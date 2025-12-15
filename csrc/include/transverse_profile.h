#ifndef TRANSVERSE_PROFILE_H
#define TRANSVERSE_PROFILE_H

#include "base.h"

/*
 * Concrete transverse profiles (beam-frame coordinates r' = (x', y', z')):
 *
 *  - PlaneWaveProfile: envelope = 1.0, phase = 0.0
 *  - HermiteTransverse: Hermite-Gaussian HG_{l,m} envelope + phase (curvature + Gouy)
 *  - GaussianTransverse: convenience wrapper for HG_{0,0}
 *
 * Units:
 *  - r' in µm
 *  - wavelength in µm
 *  - phase is dimensionless (radians)
 */

/* ===================== Plane wave transverse ===================== */

typedef struct {
    TransverseProfile base;  /* must be first */
} PlaneWaveProfile;

void PlaneWaveProfile_init(PlaneWaveProfile *p);

/* ===================== Hermite-Gaussian transverse ===================== */

typedef struct {
    TransverseProfile base;  /* must be first */

    double w0_um;  /* waist at focus [µm] */
    double zf_um;  /* focus position along z' [µm] */
    int l;         /* Hermite index in x' */
    int m;         /* Hermite index in y' */
} HermiteTransverse;

void HermiteTransverse_init(HermiteTransverse *h,
                            double w0_um,
                            double zf_um,
                            int l,
                            int m);

/* ===================== Fundamental Gaussian (HG_00) ===================== */

typedef HermiteTransverse GaussianTransverse;

void GaussianTransverse_init(GaussianTransverse *g,
                             double w0_um,
                             double zf_um);

#endif /* TRANSVERSE_PROFILE_H */

