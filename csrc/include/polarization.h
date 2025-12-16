/* csrc/include/polarization.h */
#ifndef POLARIZATION_H
#define POLARIZATION_H

#include <stdbool.h>

/*
 * Polarization (Jones-like):
 *
 *   p = p1 * e1 + p2 * exp(i*delta) * e2
 *
 * vector(phase) returns the real polarization vector:
 *   Re[ (p1 e1 + p2 e^{i delta} e2) e^{i phase} ]
 * = p1 e1 cos(phase) + p2 e2 cos(phase + delta)
 */

typedef struct
{
    double e1[3]; /* unit vector */
    double e2[3]; /* unit vector */
    double p1;    /* amplitude along e1 */
    double p2;    /* amplitude along e2 */
    double delta; /* phase lag (radians) of e2 relative to e1 */
} Polarization;

/* Basic constructor (normalizes e1,e2). Returns 0 on success. */
int Polarization_init(Polarization *p,
                      const double e1[3],
                      const double e2[3],
                      double p1,
                      double p2,
                      double delta);

/* Real polarization vector at carrier phase. */
void Polarization_vector(const Polarization *p,
                         double phase,
                         double out_vec[3]);

/* p2 ~ 0 => effectively linear along e1. */
bool Polarization_is_pure_linear(const Polarization *p, double tol);

/*
 * Build a transverse orthonormal basis (e1,e2) from k_vec, tied to the lab frame
 * (using z as reference when possible, otherwise x), then rotate it by angle_deg
 * in the transverse plane around k_hat.
 *
 * This function does NOT set p1/p2/delta; it only returns e1/e2 so you can call
 * Polarization_init(...) for general Jones-based polarization.
 *
 * Returns 0 on success.
 */
int Polarization_make_basis_angle(const double k_vec[3],
                                  double angle_deg,
                                  double e1_out[3],
                                  double e2_out[3]);

/*
 * from_lab_direction:
 *   - k_vec: propagation direction (lab)
 *   - pol_hint: desired polarization direction (lab), projected onto plane ⟂ k
 *   - delta: phase lag (radians)
 * Returns 0 on success.
 */
int Polarization_from_lab_direction(Polarization *p,
                                    const double k_vec[3],
                                    const double pol_hint[3],
                                    double delta);

/*
 * LinearPolarization equivalent (convenience):
 *   Builds transverse basis (e1,e2), rotates it by angle_deg, then sets
 *   p1=1, p2=0, delta=0.
 *
 * angle_deg in degrees.
 */
int LinearPolarization_init(Polarization *p,
                            const double k_vec[3],
                            double angle_deg);

/*
 * CircularPolarization equivalent (convenience):
 *   Builds transverse basis (e1,e2), rotates it by angle_deg, then sets
 *   p1=p2=1 and delta=±pi/2 depending on sense.
 *
 * sense strings accepted:
 *   "right","cw","clockwise","left","ccw","counterclockwise","counter-clockwise".
 */
int CircularPolarization_init(Polarization *p,
                              const double k_vec[3],
                              const char *sense,
                              double angle_deg);

#endif /* POLARIZATION_H */
