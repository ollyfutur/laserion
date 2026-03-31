#ifndef LASER_H
#define LASER_H

#include <stddef.h>
#include "base.h"
#include "temporal_profile.h"
#include "transverse_profile.h"
#include "polarization.h"

/*
 * SinglePulse:
 *   E(t, r) = E0 * temporal(t_eff) * transverse(r_beam) * pol_vector(phase)
 *
 * where:
 *   t_eff = t - z'/c   if use_retarded_time
 *   z'    = (r - r_start) · k_hat
 *
 * phase = omega * t_eff + transverse_phase(r_beam) + phase_0
 *
 * Units:
 *   - r in µm
 *   - t in fs
 *   - wavelength in µm
 *   - c in µm/fs (0.299792458)
 *   - E0 whatever unit you use consistently (GV/m in your Python)
 */

typedef struct
{
    LaserPulse base; /* must be first */

    double E0;
    double wavelength_um;
    double phase0;

    double k_hat[3];
    double r_start[3];

    int use_retarded_time;

    /* beam-frame transverse basis vectors in lab coordinates */
    double e_xb[3];
    double e_yb[3];

    double c_um_per_fs;
    double omega_rad_per_fs;

    /* composed sub-objects (non-owning pointers + value polarization) */
    const TemporalProfile *temporal;
    const TransverseProfile *transverse;
    Polarization polarization;

    /* Optional A(t,r) integration parameters (simple trapezoid rule) */
    int A_enabled;
    double A_tmin_fs;
    double A_tmax_fs;
    double A_dt_fs;

    /*
     * Maxwell correction: adds the longitudinal field component E_∥ along k_hat
     * required by Gauss's law (∇·E = 0) in the paraxial approximation:
     *
     *   E_∥ = -(1/k) ∇_⊥ · E_⊥^(q)
     *
     * where E_⊥^(q) is the transverse field with carrier phase shifted by π/2
     * (quadrature component), and ∇_⊥ is computed via central finite differences
     * with step size maxwell_fd_h_um (0 = auto: wavelength/100).
     */
    int use_maxwell_correction;
    double maxwell_fd_h_um; /* 0 = auto */
} SinglePulse;

/* Initialise SinglePulse. Returns 0 on success. */
int SinglePulse_init(SinglePulse *p,
                     double E0,
                     double wavelength_um,
                     const TemporalProfile *temporal,
                     const TransverseProfile *transverse,
                     const Polarization *polarization,
                     double phase0,
                     const double k_vec[3],
                     int use_retarded_time,
                     const double r_start[3]);

/* Enable/define parameters for A(t,r) numerical integration. */
void SinglePulse_enable_A(SinglePulse *p, double tmin_fs, double tmax_fs, double dt_fs);

/* Enable Maxwell correction. fd_h_um: step size for finite differences (0 = auto). */
void SinglePulse_enable_maxwell_correction(SinglePulse *p, double fd_h_um);

/* MultiPulse: sum of multiple LaserPulse* */
typedef struct
{
    LaserPulse base; /* must be first */
    const LaserPulse **pulses;
    size_t count;

    int A_enabled;
    double A_tmin_fs;
    double A_tmax_fs;
    double A_dt_fs;
} MultiPulse;

/* MultiPulse init: pulses is an array of LaserPulse* of length count. */
int MultiPulse_init(MultiPulse *m, const LaserPulse **pulses, size_t count);

/* MultiPulse init: pulses is an array of LaserPulse* of length count. */
void MultiPulse_enable_A(MultiPulse *m, double tmin_fs, double tmax_fs, double dt_fs);

#endif /* LASER_H */
