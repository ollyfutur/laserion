import numpy as np

# %% ################## ####
#### Polarization class ####
#### ################## ####


class Polarization:
    """
    General polarization state using a simple Jones-style description:

        p = p1 * e1 + p2 * exp(i * delta) * e2

    where e1, e2 are orthonormal basis vectors (in 3D),
    p1, p2 are real amplitudes, and delta is the phase lag of e2 relative to e1.
    """

    def __init__(self, e1, e2, p1=1.0, p2=0.0, delta=0.0):
        e1 = np.asarray(e1, dtype=float)
        e2 = np.asarray(e2, dtype=float)

        # normalize & (assume) orthogonal
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = e2 / np.linalg.norm(e2)

        self.p1 = float(p1)
        self.p2 = float(p2)
        self.delta = float(delta)

    def vector(self, phase: float) -> np.ndarray:
        """
        Return the *real* polarization vector at given carrier phase.

        E ∝ Re[ (p1 e1 + p2 e^{i δ} e2) e^{i phase} ].
        """
        c = np.exp(1j * phase)  # carrier
        v_complex = self.p1 * self.e1 * c + self.p2 * np.exp(1j * self.delta) * self.e2 * c
        return np.real(v_complex)

    def is_pure_linear(self, tol=1e-12) -> bool:
        """
        Rough test: p2 ≈ 0 => effectively linear along e1.
        Useful if you want to restrict some approximations to linear pol.
        """
        return abs(self.p2) < tol

    @classmethod
    def from_lab_direction(cls, k_vec, pol_hint, delta=0.0):
        """
        Build a Polarization object for a beam with propagation k_vec (lab),
        using a lab-frame 'hint' pol_hint (e.g. (1,0,0) for 'x-like').

        The resulting polarization is strictly transverse to k̂ and uses a
        lab-consistent basis (e1,e2) derived from k̂ and the lab z-axis.

        Parameters
        ----------
        k_vec : array_like, shape (3,)
            Propagation direction in lab frame.
        pol_hint : array_like, shape (3,)
            Desired polarization direction in lab frame (will be projected
            onto the transverse plane).
        delta : float
            Phase lag between e1 and e2 (for elliptical / circular states).
            For linear polarization, use delta = 0.
        """
        k_vec = np.asarray(k_vec, dtype=float)
        k_hat = k_vec / np.linalg.norm(k_vec)

        # --- build beam-frame transverse basis (e1,e2) ---
        z_hat = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_hat, k_hat)) < 0.999999:
            e1 = z_hat - np.dot(z_hat, k_hat) * k_hat
            e1 /= np.linalg.norm(e1)
        else:
            e1 = np.array([1.0, 0.0, 0.0])  # fallback if k ∥ z

        e2 = np.cross(k_hat, e1)
        e2 /= np.linalg.norm(e2)

        # --- project pol_hint onto transverse plane ---
        ph = np.asarray(pol_hint, dtype=float)
        if np.linalg.norm(ph) == 0:
            raise ValueError("pol_hint cannot be zero.")

        ph_perp = ph - np.dot(ph, k_hat) * k_hat
        if np.linalg.norm(ph_perp) < 1e-12:
            # Nearly parallel to k: fall back to e1
            ph_perp = e1.copy()
        ph_perp /= np.linalg.norm(ph_perp)

        # components in (e1,e2)
        p1 = float(np.dot(ph_perp, e1))
        p2 = float(np.dot(ph_perp, e2))
        norm_p = np.hypot(p1, p2)
        if norm_p > 0:
            p1 /= norm_p
            p2 /= norm_p

        return cls(e1=e1, e2=e2, p1=p1, p2=p2, delta=delta)


class LinearPolarization(Polarization):
    """
    Linear polarization in the transverse plane of a beam.

    For a beam with propagation direction k_vec, we build a transverse basis
    (e1_base, e2_base) tied to the lab frame (using z as a reference if possible).
    The polarization is then:

        E ∝ cos(angle) * e1_base + sin(angle) * e2_base,

    with 'angle' measured in that transverse plane. For k_vec || z, e1_base = x,
    e2_base = y, so angle=0 → x, angle=90° → y, as in the standard case.
    """

    def __init__(self, k_vec, angle: float = 0.0):
        k_vec = np.asarray(k_vec, dtype=float)
        k_hat = k_vec / np.linalg.norm(k_vec)

        # Build transverse basis (e1_base, e2_base) tied to lab frame
        z_hat = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_hat, k_hat)) < 0.999999:
            e1_base = z_hat - np.dot(z_hat, k_hat) * k_hat
            e1_base /= np.linalg.norm(e1_base)
        else:
            # If k is (anti-)parallel to z, fall back to x as reference
            e1_base = np.array([1.0, 0.0, 0.0])

        e2_base = np.cross(k_hat, e1_base)
        e2_base /= np.linalg.norm(e2_base)

        # Angle in radians
        theta = angle * np.pi / 180.0

        # Polarization direction in the transverse plane
        # e_pol = cos θ e1_base + sin θ e2_base
        p1 = np.cos(theta)
        p2 = np.sin(theta)

        super().__init__(e1=e1_base, e2=e2_base, p1=p1, p2=p2, delta=0.0)


class CircularPolarization(Polarization):
    """
    Circular polarization in the transverse plane of a beam.

    For a beam with propagation direction k_vec, we build a transverse basis
    (e1_base, e2_base) tied to the lab frame (using z as a reference if possible).

    We then optionally rotate this basis by 'angle' in the transverse plane:

        e1 = cos(angle) * e1_base + sin(angle) * e2_base
        e2 = -sin(angle) * e1_base + cos(angle) * e2_base

    and define circular polarization with equal amplitudes p1 = p2 = 1 and
    phase lag δ = ±π/2 between e1 and e2.
    """

    def __init__(self, k_vec, sense: str = "right", angle: float = 0.0):
        k_vec = np.asarray(k_vec, dtype=float)
        k_hat = k_vec / np.linalg.norm(k_vec)

        # Build transverse basis (e1_base, e2_base) tied to lab frame
        z_hat = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_hat, k_hat)) < 0.999999:
            e1_base = z_hat - np.dot(z_hat, k_hat) * k_hat
            e1_base /= np.linalg.norm(e1_base)
        else:
            e1_base = np.array([1.0, 0.0, 0.0])

        e2_base = np.cross(k_hat, e1_base)
        e2_base /= np.linalg.norm(e2_base)

        # Rotate the transverse basis by 'angle' in that plane
        theta = angle * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        e1 = cos_t * e1_base + sin_t * e2_base
        e2 = -sin_t * e1_base + cos_t * e2_base

        # Choose phase lag for handedness
        sense_l = sense.lower()
        if sense_l in ("right", "cw", "clockwise"):
            delta = +np.pi / 2.0
        elif sense_l in ("left", "ccw", "counterclockwise", "counter-clockwise"):
            delta = -np.pi / 2.0
        else:
            raise ValueError(f"Unknown circular polarization sense: {sense!r}")

        # Circular: equal amplitudes along e1, e2, ±π/2 phase lag
        super().__init__(e1=e1, e2=e2, p1=1.0, p2=1.0, delta=delta)
