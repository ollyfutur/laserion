# %%
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf, gamma
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from typing import Optional, Sequence
import matplotlib.pyplot as plt

# %% ################### ####
#### Intialize templates ####
#### ################### ####


class LaserPulse(ABC):
    """
    Abstract base class for any laser pulse.
    Defines the interface that all specific pulses must implement.
    """

    @abstractmethod
    def E(self, t, r: np.ndarray) -> np.ndarray:
        """
        Electric field vector at time t and position r.
        Parameters
        ----------
        t : float
            Time [fs]
        r : array_like, shape (3,)
            Spatial position (x, y, z) [µm]
        Returns
        -------
        np.array([Ex, Ey, Ez])
        """
        pass

    @abstractmethod
    def A(self, t, r: np.ndarray) -> np.ndarray:
        """
        Vector potential at time t and position r.
        Must satisfy E = -dA/dt.

        Returns np.array([Ax, Ay, Az]).
        """
        pass


class TransverseProfile(ABC):
    @abstractmethod
    def __call__(self, r: np.ndarray, wavelength: np.ndarray) -> float:
        """Return transverse envelope at position r = (x, y, z)."""
        pass

    def phase(self, r: np.ndarray, wavelength: np.ndarray) -> float:
        """
        Additional spatial phase φ_profile(r) to be added to the carrier.

        Default: no extra phase (plane wave, etc.).
        Override in subclasses (e.g. Hermite-Gaussian) if needed.
        """
        return 0.0


class TemporalProfile(ABC):
    @abstractmethod
    def __call__(self, t: float) -> float:
        """Return temporal envelope at time t."""
        pass


class IonizationModel(ABC):
    @abstractmethod
    def rate(self, E_abs: float, species: str, Z: int) -> float:
        """
        Ionization rate w(|E|) [1/fs] for given species and final charge state Z.
        """
        pass


# %% ############################### ####
#### Options for transverse profiles ####
#### ############################### ####


class PlaneWaveProfile(TransverseProfile):
    def __call__(self, r, wavelength: Optional[float] = None):
        # no transverse dependence
        return 1.0


class HermiteTransverse(TransverseProfile):
    def __init__(self, w0, zf=0.0, l=0, m=0, x0=0.0, y0=0.0):
        """
        Hermite-Gaussian transverse profile (envelope only), in the *beam frame*.

        Beam frame coordinates: r' = (x', y', z'), where z' is along the
        propagation direction k̂, and the origin is at r_start.

        w0 : float
            Beam waist at focus [µm]
        zf : float
            Focus position along propagation axis z' [µm], measured from r_start.
        l, m : int
            Hermite mode indices in x' and y'
        x0, y0 : float
            Transverse offset of the beam centre in the beam frame [µm].
        """

        self.w0 = w0
        self.zf = zf
        self.l = l
        self.m = m
        self.x0 = x0
        self.y0 = y0

    def _compute_params(self, wavelength: float):
        z0 = np.pi * self.w0**2 / wavelength
        k = 2 * np.pi / wavelength
        return z0, k

    def __call__(self, r: np.ndarray, wavelength: float):
        # r is the beam-frame coords: (x', y', z')
        x, y, z = np.asarray(r, dtype=float)
        x_rel = x - self.x0
        y_rel = y - self.y0
        z_rel = z - self.zf  # shift to focus position

        z0, _ = self._compute_params(wavelength)
        wz = self.w0 * np.sqrt(1.0 + (z_rel / z0) ** 2)

        # Hermite polynomials H_l, H_m
        H_l = np.polynomial.hermite.hermval(np.sqrt(2) * x_rel / wz, [0] * self.l + [1])
        H_m = np.polynomial.hermite.hermval(np.sqrt(2) * y_rel / wz, [0] * self.m + [1])

        gauss = np.exp(-(x_rel**2 + y_rel**2) / wz**2)

        # transverse envelope
        return (self.w0 / wz) * H_l * H_m * gauss

    def phase(self, r: np.ndarray, wavelength: float) -> float:
        """
        Spatial phase for Hermite-Gaussian mode: curvature + Gouy phase,
        in beam-frame coordinates, centred on (x0, y0, zf).
        """
        x, y, z = np.asarray(r, dtype=float)
        x_rel = x - self.x0
        y_rel = y - self.y0
        z_rel = z - self.zf  # shift to focus position
        z0, k = self._compute_params(wavelength)

        # Radius of curvature R(z)
        if z_rel == 0.0:
            curvature = 0.0
        else:
            R = (z_rel**2 + z0**2) / z_rel
            curvature = -k * (x_rel**2 + y_rel**2) / (2.0 * R)

        # Gouy phase
        zeta = np.arctan2(z_rel, z0)
        gouy = (self.l + self.m + 1) * zeta

        return curvature + gouy


class GaussianTransverse(HermiteTransverse):
    """
    Fundamental Gaussian transverse profile (HG_00),
    implemented as HermiteTransverse with l = m = 0.
    """

    def __init__(self, w0, zf=0.0, x0=0.0, y0=0.0):
        """
        w0 : float
            Beam waist at focus [µm]
        zf : float
            Focus position along beam axis z' [µm] from r_start.
        x0, y0 : float
            Transverse offset of the beam centre in beam-frame coords [µm].
        """
        super().__init__(w0=w0, zf=zf, l=0, m=0, x0=x0, y0=y0)


# %% ############################# ####
#### Options for temporal profiles ####
#### ############################# ####


class GaussianTemporal(TemporalProfile):
    def __init__(self, tau):
        """
        Gaussian temporal profile:
            envelope(t) = exp(-2 t^2 / tau^2)
        tau : float
            RMS pulse duration [fs]
        """
        self.tau = tau

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-2.0 * t**2 / self.tau**2)


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


# %% ################################################### ####
#### Class that combines the profiles into a laser pulse ####
#### ################################################### ####


class SinglePulse(LaserPulse):
    """
    General laser pulse with factored structure:
        E(t, r) = E_0 * temporal(t_eff) * transverse(r) * cos(total_phase(t, r)) * polarization

    - temporal  : TemporalProfile instance (e.g. GaussianTemporal)
    - transverse: TransverseProfile instance (e.g. PlaneWaveProfile, GaussianTransverse, HermiteTransverse)
    - polarization: Polarization instance (e.g. LinearPolarization, CircularPolarization, or Polarization)"
    """

    def __init__(self, E_0, wavelength, temporal: TemporalProfile, transverse: TransverseProfile, polarization: Polarization, phase_0=0.0, k_vec=(0.0, 0.0, 1.0), use_retarded_time=True, r_start=(0.0, 0.0, 0.0)):
        """
        Parameters
        ----------
        E_0 : float
            Peak electric field amplitude [GV/m].
        wavelength : float
            Laser wavelength [µm].
        temporal : TemporalProfile
            Temporal envelope object, e.g. GaussianTemporal(tau).
        transverse : TransverseProfile
            Transverse envelope object (GaussianTransverse, HermiteTransverse, ...),
            defined in beam-frame coordinates.
        polarization : Polarization
            Polarization object defining the polarization state
            in the beam frame.
        phase_0 : float
            Global carrier phase offset [rad].
        k_vec : tuple of 3 floats
            Propagation direction in lab frame (will be normalized).
        use_retarded_time : bool
            If True, the envelope uses t_eff = t - z'/c with z' along k̂
            and origin at r_start.
        r_start : 3-tuple
            Lab position of the pulse envelope centre at t=0 (z' = 0).
        """

        if not isinstance(temporal, TemporalProfile):
            raise TypeError("temporal must be an instance of TemporalProfile.")

        if not isinstance(transverse, TransverseProfile):
            raise TypeError("transverse must be an instance of TransverseProfile.")

        if not isinstance(polarization, Polarization):
            raise TypeError("polarization must be an instance of Polarization.")

        self.E_0 = E_0
        self.wavelength = wavelength
        self.temporal = temporal
        self.transverse = transverse
        self.phase_0 = phase_0
        self.polarization = polarization

        k_vec = np.array(k_vec, dtype=float)
        self.k_hat = k_vec / np.linalg.norm(k_vec)

        self.use_retarded_time = use_retarded_time
        self.r_start = np.asarray(r_start, dtype=float)

        # constants in µm/fs units
        self.c = 0.299792458
        self.omega = 2.0 * np.pi * self.c / wavelength
        self.k = 2.0 * np.pi / wavelength

        # -------------------------------------------------
        # Beam-frame transverse basis tied to the lab frame
        # -------------------------------------------------
        z_hat = np.array([0.0, 0.0, 1.0])
        # If k_hat not parallel to z, project z onto transverse plane
        if abs(np.dot(z_hat, self.k_hat)) < 0.999999:
            e1 = z_hat - np.dot(z_hat, self.k_hat) * self.k_hat
            e1 /= np.linalg.norm(e1)
        else:
            # If beam is (anti-)parallel to z, fall back to x as reference
            e1 = np.array([1.0, 0.0, 0.0])

        e2 = np.cross(self.k_hat, e1)
        e2 /= np.linalg.norm(e2)

        # Save beam-frame transverse basis
        self.e_xb = e1  # x' direction in lab coords
        self.e_yb = e2  # y' direction in lab coords
        # -------------------------------------------------

        self._A_cache = None

    def _axis_coords(self, r: np.ndarray):
        """
        Return (dr, z') with
          dr = r - r_start  (vector in lab frame),
          z' = dr · k_hat   (coordinate along beam axis).
        """
        dr = np.asarray(r, dtype=float) - self.r_start
        z_prime = float(np.dot(dr, self.k_hat))
        return dr, z_prime

    def _to_beam_frame(self, r: np.ndarray) -> np.ndarray:
        """
        Convert lab position r to beam-frame coordinates (x', y', z'):

            dr = r - r_start
            x' = dr · e_xb
            y' = dr · e_yb
            z' = dr · k_hat
        """
        dr, z_prime = self._axis_coords(r)
        x_prime = float(np.dot(dr, self.e_xb))
        y_prime = float(np.dot(dr, self.e_yb))
        return np.array([x_prime, y_prime, z_prime])

    def _t_eff(self, t, r: np.ndarray) -> float:
        """
        Effective (retarded) time at position r.

        If use_retarded_time:
            z' = (r - r_start) · k_hat
            t_eff = t - z'/c
        else:
            t_eff = t
        """
        if not self.use_retarded_time:
            return float(t)

        _, z_prime = self._axis_coords(r)
        return float(t) - z_prime / self.c

    def _spatial_phase(self, r_beam: np.ndarray) -> float:
        """
        Spatial phase contribution (excluding ω t_eff):

            φ_spatial = φ_profile(r_beam) + phase_0

        The plane-wave term −k z' is effectively taken into account through
        the retarded time t_eff = t − z'/c in the carrier cos(ω t_eff + φ_spatial).
        """
        phi_profile = self.transverse.phase(r_beam, self.wavelength)
        return phi_profile + self.phase_0

    def E(self, t, r: np.ndarray) -> np.ndarray:
        """
        Electric field:
            E = E_0 * temporal(t_eff) * transverse(r_beam)
                * cos(ω t_eff + φ_spatial(r_beam)) * polarization_vector
        """
        r = np.asarray(r, dtype=float)
        r_beam = self._to_beam_frame(r)

        t_eff = self._t_eff(t, r)

        env_t = self.temporal(t_eff)
        env_r = self.transverse(r_beam, self.wavelength)

        phi_spatial = self._spatial_phase(r_beam)

        phase = self.omega * t_eff + phi_spatial
        pol_vec = self.polarization.vector(phase)

        return self.E_0 * env_t * env_r * pol_vec

    def build_A_cache(self, r, t_min=None, t_max=None, dt=None, envelope_cut=1e-8):
        """
        Precompute A(t, r) = -∫ E(t', r) dt' on a time grid,
        and store an interpolator for later calls to A(t, r).

        If t_min/t_max are None and the temporal profile is GaussianTemporal,
        the window is chosen so that the envelope at this position is
        ~ envelope_cut at the edges.
        """
        r = np.asarray(r, dtype=float)

        # --- automatic t_min / t_max when GaussianTemporal ---
        if (t_min is None or t_max is None) and isinstance(self.temporal, GaussianTemporal):
            tau = self.temporal.tau
            # envelope_cut = exp(-2 t_eff^2 / tau^2)
            t_lim = tau * np.sqrt(0.5 * np.log(1.0 / envelope_cut))

            # pulse arrival time at this position
            if self.use_retarded_time:
                _, z_prime = self._axis_coords(r)
                t_center = z_prime / self.c
            else:
                t_center = 0.0

            if t_min is None:
                t_min = t_center - t_lim
            if t_max is None:
                t_max = t_center + t_lim

        if t_min is None or t_max is None:
            raise ValueError("t_min and t_max must be provided for non-Gaussian temporal profiles.")

        # --- default dt: laser period / 150 ---
        if dt is None:
            T_laser = self.wavelength / self.c
            dt = T_laser / 150.0

        # time grid
        n_steps = int(np.ceil((t_max - t_min) / dt)) + 1
        t_grid = t_min + dt * np.arange(n_steps)

        # compute E(t, r) on the grid
        E_grid = np.array([self.E(t, r) for t in t_grid])

        # A(t) = -∫ E dt
        A_grid = -cumulative_trapezoid(E_grid, t_grid, axis=0, initial=0.0)

        interp = interp1d(
            t_grid,
            A_grid,
            axis=0,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self._A_cache = {
            "r": r,
            "t_grid": t_grid,
            "A_grid": A_grid,
            "interp": interp,
        }

    def A(self, t, r: np.ndarray) -> np.ndarray:
        """
        Vector potential A(t, r), computed numerically from E.

        - If no cache exists, or the cached position differs from r,
        build a new cache at this r using build_A_cache(r, ...).
        - Then interpolate from that cache.

        This is efficient if you call A many times at the same r; if you
        change r often, you pay for a new time-grid integration each time.
        """
        r = np.asarray(r, dtype=float)

        # Check if we need to (re)build cache
        need_new_cache = self._A_cache is None or not np.allclose(r, self._A_cache["r"], rtol=1e-12, atol=1e-12)

        if need_new_cache:
            # Automatic t_min/t_max/dt for GaussianTemporal, or explicit if you prefer
            self.build_A_cache(r)

        interp = self._A_cache["interp"]
        return interp(t)

    def plot(self, component: str, plane: str, *, t: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, xrange=(-5.0, 5.0), yrange=(-5.0, 5.0), N: int = 400, r_for_A=None):
        """
        Plot electric or vector potential field components.

        Parameters
        ----------
        component : str
            One of "Ex","Ey","Ez","Ax","Ay","Az".
        plane : str
            1D:
                "t"  → component vs t (x,y,z fixed)
                "x"  → vs x (t,y,z fixed)
                "y"  → vs y (t,x,z fixed)
                "z"  → vs z (t,x,y fixed)
            2D:
                any 2-letter combo of "x","y","z", e.g.
                "xy","yx","xz","zx","yz","zy".
        t, x, y, z : float
            Coordinates to hold fixed (depending on plane).
        xrange, yrange : tuple
            For 1D: xrange is the range of the *variable* axis.
            For 2D: xrange is the horizontal axis range, yrange the vertical.
        N : int
            Number of samples (per axis).
        r_for_A : ndarray or None
            Position where A(t,r) has been cached. If None and A is requested,
            A is cached at (x,y,z).
        """

        # ---------- Determine if E or A ----------
        if component.startswith("E"):
            field_fn = lambda tt, rr: self.E(tt, rr)
            idx = {"Ex": 0, "Ey": 1, "Ez": 2}[component]
        elif component.startswith("A"):
            # For A, our implementation assumes A is precomputed at a single r.
            rr_cache = np.array([x, y, z]) if r_for_A is None else np.asarray(r_for_A)
            self.build_A_cache(rr_cache)
            field_fn = lambda tt, rr: self.A(tt, rr_cache)
            idx = {"Ax": 0, "Ay": 1, "Az": 2}[component]
        else:
            raise ValueError("component must be one of Ex,Ey,Ez,Ax,Ay,Az")

        # =================================================
        #            1D plots: "t", "x", "y", "z"
        # =================================================
        if plane in ("t", "x", "y", "z"):
            # variable axis grid always taken from xrange
            var = np.linspace(*xrange, N)
            vals = np.zeros_like(var)

            if plane == "t":
                # vary time, keep spatial position fixed
                r_vec = np.array([x, y, z], dtype=float)
                for i, tt in enumerate(var):
                    vals[i] = field_fn(tt, r_vec)[idx]
                xlabel = "t [fs]"
            else:
                # vary one spatial coordinate, keep t and others fixed
                axis_map = {"x": 0, "y": 1, "z": 2}
                ax_idx = axis_map[plane]
                r0 = np.array([x, y, z], dtype=float)

                for i, coord in enumerate(var):
                    r_vec = r0.copy()
                    r_vec[ax_idx] = coord
                    vals[i] = field_fn(t, r_vec)[idx]

                xlabel = f"{plane} [µm]"

            plt.figure(figsize=(7, 4))
            plt.plot(var, vals)
            plt.title(f"{component} vs {plane}")
            plt.xlabel(xlabel)
            plt.ylabel(component)
            plt.grid(True)
            plt.show()
            return

        # =================================================
        #      2D MAPS: "xy", "yx", "xz", "zx", "yz", "zy"
        # =================================================
        if len(plane) != 2 or any(ax not in "xyz" for ax in plane):
            raise ValueError("plane must be 't','x','y','z' or any 2-letter combo of 'x','y','z' " "(e.g. 'xy','xz','yz','zx','yx','zy').")

        # map 'x','y','z' → index 0,1,2
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax1 = axis_map[plane[0]]  # horizontal axis
        ax2 = axis_map[plane[1]]  # vertical axis

        # base position r0: start from given (x,y,z)
        r0 = np.array([x, y, z], dtype=float)

        # set up coordinate ranges for the two varying axes
        X = np.linspace(*xrange, N)  # horizontal
        Y = np.linspace(*yrange, N)  # vertical
        XX, YY = np.meshgrid(X, Y)

        # build field map
        vals = np.zeros_like(XX)
        for i in range(N):
            for j in range(N):
                r_vec = r0.copy()
                r_vec[ax1] = XX[i, j]
                r_vec[ax2] = YY[i, j]
                vals[i, j] = field_fn(t, r_vec)[idx]

        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            vals,
            extent=(xrange[0], xrange[1], yrange[0], yrange[1]),
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im, label=component)
        plt.title(f"{component} in {plane}-plane at t={t}")
        plt.xlabel(plane[0] + " [µm]")
        plt.ylabel(plane[1] + " [µm]")
        plt.show()


class MultiPulse(LaserPulse):
    """
    Superposition of several LaserPulse objects.
    E_total(t, r) = sum_i E_i(t, r)
    A_total(t, r) is obtained numerically from E_total.

    Sub-pulses can have different wavelengths, phases, polarizations, etc.
    """

    def __init__(self, pulses):
        """
        Parameters
        ----------
        pulses : sequence of LaserPulse
            List/tuple of pulse objects (e.g. SinglePulse instances).
        """
        self.pulses = list(pulses)
        for p in self.pulses:
            if not isinstance(p, LaserPulse):
                raise TypeError("All entries in pulses must be instances of LaserPulse.")

        # use c from first pulse if available, otherwise default
        first = self.pulses[0]
        self.c = getattr(first, "c", 0.299792458)

        # cache for A(t, r)
        self._A_cache = None

    # ---------- basic field ----------

    def E(self, t, r: np.ndarray) -> np.ndarray:
        """
        Total electric field: sum of sub-pulse fields.
        """
        r = np.asarray(r, dtype=float)
        E_tot = np.zeros(3, dtype=float)
        for p in self.pulses:
            E_tot += p.E(t, r)
        return E_tot

    # ---------- A cache construction ----------

    def build_A_cache(self, r, t_min=None, t_max=None, dt=None, envelope_cut=1e-6):
        """
        Precompute A_total(t, r) = -∫ E_total(t', r) dt' on a time grid,
        and store an interpolator for later calls to A(t, r).

        If t_min/t_max are None AND all sub-pulses have GaussianTemporal,
        the window is chosen to cover all pulses at this position down to
        envelope_cut.
        """
        r = np.asarray(r, dtype=float)

        # --- automatic t_min/t_max if all temporals are GaussianTemporal ---
        if t_min is None or t_max is None:
            all_gauss = all(hasattr(p, "temporal") and isinstance(p.temporal, GaussianTemporal) for p in self.pulses)
            if all_gauss:
                t_mins = []
                t_maxs = []
                for p in self.pulses:
                    tau = p.temporal.tau
                    t_lim = tau * np.sqrt(0.5 * np.log(1.0 / envelope_cut))

                    if getattr(p, "use_retarded_time", False):
                        # use each pulse's own beam-frame geometry
                        if hasattr(p, "_axis_coords"):
                            _, z_prime = p._axis_coords(r)
                            t_center = z_prime / p.c
                        else:
                            t_center = 0.0
                    else:
                        t_center = 0.0

                    t_mins.append(t_center - t_lim)
                    t_maxs.append(t_center + t_lim)

                if t_min is None:
                    t_min = min(t_mins)
                if t_max is None:
                    t_max = max(t_maxs)

        if t_min is None or t_max is None:
            raise ValueError("t_min and t_max must be provided if not all sub-pulses use GaussianTemporal.")

        # --- default dt: use smallest wavelength among pulses, period/150 ---
        if dt is None:
            lambdas = []
            for p in self.pulses:
                if hasattr(p, "wavelength"):
                    lambdas.append(p.wavelength)
            if len(lambdas) == 0:
                raise ValueError("dt must be provided if sub-pulses lack 'wavelength' attribute.")
            lambda_min = min(lambdas)
            T_min = lambda_min / self.c
            dt = T_min / 150.0

        # time grid
        n_steps = int(np.ceil((t_max - t_min) / dt)) + 1
        t_grid = t_min + dt * np.arange(n_steps)

        # total E(t, r) on grid
        E_grid = np.array([self.E(t, r) for t in t_grid])

        # A_total(t) = -∫ E_total dt
        A_grid = -cumulative_trapezoid(E_grid, t_grid, axis=0, initial=0.0)

        interp = interp1d(
            t_grid,
            A_grid,
            axis=0,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self._A_cache = {
            "r": r,
            "t_grid": t_grid,
            "A_grid": A_grid,
            "interp": interp,
        }

    # ---------- A with cache ----------

    def A(self, t, r: np.ndarray) -> np.ndarray:
        """
        Vector potential A_total(t, r), computed from E_total.
        Lazy behavior:
          - if no cache exists or r changed, rebuild at that r with defaults.
          - then interpolate.
        """
        r = np.asarray(r, dtype=float)

        need_new = self._A_cache is None or not np.allclose(r, self._A_cache["r"], rtol=1e-12, atol=1e-12)

        if need_new:
            self.build_A_cache(r)

        interp = self._A_cache["interp"]
        return interp(t)

    # ===============================
    #      Plotting Interface
    # ===============================
    def plot(self, component: str, plane: str, *, t: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, xrange=(-5.0, 5.0), yrange=(-5.0, 5.0), N: int = 400, r_for_A=None, show_total: bool = True, show_pulses: bool = False, labels: Optional[Sequence[str]] = None, cmap: str = "viridis"):
        """
        Plot electric or vector potential field components for a MultiPulse.

        Parameters
        ----------
        component : str
            One of "Ex","Ey","Ez","Ax","Ay","Az".
        plane : str
            1D:
                "t"  → vs t (x,y,z fixed)
                "x"  → vs x (t,y,z fixed)
                "y"  → vs y (t,x,z fixed)
                "z"  → vs z (t,x,y fixed)
            2D:
                any 2-letter combo of "x","y","z", e.g.
                "xy","yx","xz","zx","yz","zy".
        t, x, y, z : float
            Coordinates to hold fixed (depending on plane).
        xrange, yrange : tuple
            For 1D: xrange is the range of the *variable* axis.
            For 2D: xrange is horizontal axis range, yrange vertical.
        N : int
            Number of samples (per axis).
        r_for_A : ndarray or None
            Position where A(t,r) has been cached. If None and A is requested,
            A is cached at (x,y,z).
        show_total : bool
            If True, plot the total field (sum of all pulses).
        show_pulses : bool
            If True, also plot individual sub-pulse contributions (1D only).
        labels : sequence of str, optional
            Labels for each sub-pulse when show_pulses=True.
            If None, labels like "pulse 0", "pulse 1", ... are used.
        cmap : str
            Colormap for 2D plots.
        """

        # ---------- E vs A selection ----------
        # Total field
        if component.startswith("E"):
            total_field_fn = lambda tt, rr: self.E(tt, rr)
            idx = {"Ex": 0, "Ey": 1, "Ez": 2}[component]

            # per-pulse E
            pulse_field_fns = [lambda tt, rr, p=p: p.E(tt, rr) for p in self.pulses]

        elif component.startswith("A"):
            # A is computed via caches at specific r
            rr_cache = np.array([x, y, z]) if r_for_A is None else np.asarray(r_for_A)

            # build cache for total A
            self.build_A_cache(rr_cache)
            total_field_fn = lambda tt, rr: self.A(tt, rr_cache)
            idx = {"Ax": 0, "Ay": 1, "Az": 2}[component]

            # per-pulse A: ensure each has cache at same rr_cache
            pulse_field_fns = []
            for p in self.pulses:
                if hasattr(p, "build_A_cache"):
                    p.build_A_cache(rr_cache)
                    pulse_field_fns.append(lambda tt, rr, pp=p: pp.A(tt, rr_cache))
                else:
                    # if sub-pulse has no A, approximate with zero
                    pulse_field_fns.append(lambda tt, rr: np.zeros(3))
        else:
            raise ValueError("component must be one of Ex,Ey,Ez,Ax,Ay,Az")

        # =================================================
        #            1D plots: "t", "x", "y", "z"
        # =================================================
        if plane in ("t", "x", "y", "z"):
            var = np.linspace(*xrange, N)
            total_vals = np.zeros_like(var)

            # base position r0
            r0 = np.array([x, y, z], dtype=float)

            axis_map = {"x": 0, "y": 1, "z": 2}

            if plane == "t":
                # vary time, position fixed
                for i, tt in enumerate(var):
                    total_vals[i] = total_field_fn(tt, r0)[idx]
            else:
                # vary one spatial coordinate, keep t fixed
                ax_idx = axis_map[plane]
                for i, coord in enumerate(var):
                    r_vec = r0.copy()
                    r_vec[ax_idx] = coord
                    total_vals[i] = total_field_fn(t, r_vec)[idx]

            # Prepare figure/axes
            fig, ax = plt.subplots(figsize=(7, 4))

            # Plot total
            if show_total:
                ax.plot(var, total_vals, label="total")

            # Plot individual pulses
            if show_pulses:
                if labels is not None and len(labels) != len(self.pulses):
                    raise ValueError("labels length must match number of pulses.")

                for k, f_pulse in enumerate(pulse_field_fns):
                    pulse_vals = np.zeros_like(var)
                    if plane == "t":
                        for i, tt in enumerate(var):
                            pulse_vals[i] = f_pulse(tt, r0)[idx]
                    else:
                        ax_idx = axis_map[plane]
                        for i, coord in enumerate(var):
                            r_vec = r0.copy()
                            r_vec[ax_idx] = coord
                            pulse_vals[i] = f_pulse(t, r_vec)[idx]

                    lab = labels[k] if labels is not None else f"pulse {k}"
                    ax.plot(var, pulse_vals, "--", label=lab)

            xlabel = "t [fs]" if plane == "t" else f"{plane} [µm]"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(component)
            ax.grid(True)
            ax.legend()
            ax.set_title(f"{component} vs {plane}")
            plt.show()
            return

        # =================================================
        #      2D MAPS: "xy", "yx", "xz", "zx", "yz", "zy"
        # =================================================
        if len(plane) != 2 or any(ax not in "xyz" for ax in plane):
            raise ValueError("plane must be 't','x','y','z' or any 2-letter combo of 'x','y','z' " "(e.g. 'xy','xz','yz','zx','yx','zy').")

        if show_pulses:
            raise ValueError("show_pulses=True is not supported for 2D plots.")

        axis_map = {"x": 0, "y": 1, "z": 2}
        ax1 = axis_map[plane[0]]  # horizontal
        ax2 = axis_map[plane[1]]  # vertical

        r0 = np.array([x, y, z], dtype=float)

        X = np.linspace(*xrange, N)
        Y = np.linspace(*yrange, N)
        XX, YY = np.meshgrid(X, Y)

        vals = np.zeros_like(XX)
        for i in range(N):
            for j in range(N):
                r_vec = r0.copy()
                r_vec[ax1] = XX[i, j]
                r_vec[ax2] = YY[i, j]
                vals[i, j] = total_field_fn(t, r_vec)[idx]

        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            vals,
            extent=(xrange[0], xrange[1], yrange[0], yrange[1]),
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        plt.colorbar(im, label=component)
        plt.title(f"{component} in {plane}-plane at t={t}")
        plt.xlabel(plane[0] + " [µm]")
        plt.ylabel(plane[1] + " [µm]")
        plt.show()


# %% ############### ####
#### Ionization data ####
#### ############### ####

ion_data = {
    "Ar": {"Z": list(range(1, 19)), "E": [15.7596119, 27.62967, 40.735, 59.58, 74.84, 91.290, 124.41, 143.4567, 422.60, 479.76, 540.4, 619.0, 685.5, 755.13, 855.5, 918.375, 4120.6657, 4426.22407]},
    "He": {"Z": list(range(1, 3)), "E": [24.587389011, 54.4177655282]},
    "H": {"Z": list(range(1, 2)), "E": [13.598434599702]},
    "Li": {"Z": list(range(1, 4)), "E": [5.391714996, 75.6400970, 122.45435913]},
    "Ne": {"Z": list(range(1, 11)), "E": [21.564541, 40.96297, 63.4233, 97.1900, 126.247, 157.934, 207.271, 239.0970, 1195.80784, 1362.199256]},
    "Xe": {"Z": list(range(1, 55)), "E": [12.1298437, 20.975, 31.05, 42.20, 54.1, 66.703, 91.6, 105.9778, 179.84, 202.0, 229.02, 255.0, 281, 314, 343, 374, 404, 434, 549, 582, 616, 650, 700, 736, 818, 857.0, 1493, 1571, 1653, 1742, 1826, 1919, 2023, 2113, 2209, 2300, 2556, 2637, 2726, 2811, 2975, 3068, 3243, 3333.8, 7660, 7889, 8144, 8382, 8971, 9243, 9581, 9810.37, 40271.724, 41299.892]},
    "O": {"Z": list(range(1, 9)), "E": [13.618055, 35.12112, 54.93554, 77.41350, 113.8990, 138.1189, 739.32683, 871.4099138]},
    "N": {"Z": list(range(1, 8)), "E": [14.53413, 29.60125, 47.4453, 77.4735, 97.8901, 552.06733, 667.0461377]},
}

# %% ################ ####
#### Ionization model ####
#### ################ ####


class ADKModel(IonizationModel):
    def __init__(self, ion_data=ion_data):
        self.ion_data = ion_data

    def ionization_energy(self, species: str, Z: int) -> float:
        """Return ionization energy [eV] for given species and final charge state Z."""
        try:
            levels = self.ion_data[species]
        except KeyError:
            raise ValueError(f"Unknown species: {species!r}")

        Z_list = levels["Z"]
        E_list = levels["E"]

        try:
            idx = Z_list.index(Z)
        except ValueError:
            raise ValueError(f"Species {species!r} has no entry for Z={Z}.")

        return E_list[idx]

    @staticmethod
    def _scalar_rate(E: float, ion_ene: float, Z: int) -> float:
        """ADK rate w(|E|) [1/fs] for a single level."""
        E_abs = np.abs(E)
        if E_abs <= 0:
            return 0.0

        n = 3.69 * Z / np.sqrt(ion_ene)

        C = 2 * n - 1
        A = 1.52 * 4**n * ion_ene * (20.5 * ion_ene**1.5) ** C / (n * gamma(2 * n))
        B = 6.83 * ion_ene**1.5

        return A / E_abs**C * np.exp(-B / E_abs)

    def rate(self, E_abs: float, species: str, Z: int) -> float:
        """ADK rate w(|E|) [1/fs] for given species and charge state."""
        ion_ene = self.ionization_energy(species, Z)
        return self._scalar_rate(E_abs, ion_ene, Z)


# %% ####################################################### ####
#### Calculation of the Momentum Distribution Function (MDF) ####
#### ####################################################### ####


class MDF:
    """
    Momentum Distribution Function for electrons born by ionization
    in a given laser field at a fixed position r.

    Usage
    -----
    mdf = MDF(pulse, species="He", Z=[1, 2], r=[0,0,0])
    mdf.plot("px")
    mdf.plot("pxpy")
    """

    CONV_A_TO_P = -0.0005866792097892349  # A [GV/m·fs] -> p [m_e c]

    def __init__(
        self,
        pulse,
        species: str,
        Z,
        r,
        ion_model: Optional[IonizationModel] = None,
        envelope_cut: float = 1e-6,
        dt: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        pulse : SinglePulse or MultiPulse
            Object with .E(t, r), .build_A_cache(r, ...), and ._A_cache.
        species : str
            Atomic species key in ion_data (e.g. "He", "Ar").
        Z : int or sequence of int
            Final charge state(s) after ionization.
            - int        → single level, e.g. 1
            - [1, 2, 3]  → multiple levels, all computed on the same field.
        r : array_like, shape (3,)
            Position where the ionization is evaluated [µm].
        ion_model : IonizationModel, optional
            If None: a new ADKModel() is constructed.
        envelope_cut : float, optional
            Used by pulse.build_A_cache for automatic t_min/t_max
            (for GaussianTemporal).
        dt : float, optional
            Time step for the A(t) cache; if None, pulse chooses
            a default ~ laser period / 150.
        """
        if ion_model is None:
            ion_model = ADKModel()

        self.pulse = pulse
        self.species = species
        self.ion_model = ion_model
        self.r = np.asarray(r, dtype=float)

        # Normalize Z to a list
        if np.isscalar(Z):
            self.Z_list = [int(Z)]
        else:
            self.Z_list = [int(z) for z in Z]

        # ---- 1) Build A(t,r) cache once ----
        pulse.build_A_cache(self.r, t_min=None, t_max=None, dt=dt, envelope_cut=envelope_cut)

        cache = pulse._A_cache
        self.t_grid = cache["t_grid"]  # (N,)
        A_grid = cache["A_grid"]  # (N, 3)

        # p(t) for any level is the same function of A(t)
        self.p_grid = A_grid * self.CONV_A_TO_P  # (N, 3), in units m_e c

        # ---- 2) Field magnitude |E(t,r)| (common to all Z) ----
        E_grid = np.array([pulse.E(t, self.r) for t in self.t_grid])  # (N, 3)
        self.E_abs = np.linalg.norm(E_grid, axis=1)  # (N,)

        # ---- 3) For each Z: compute w(t), S(t), dP(t), P_ion ----
        self.w_levels = []  # list of arrays: w_t[Z_i] shape (N,)
        self.S_levels = []  # survival probabilities
        self.dP_levels = []  # dP[Z_i, n]
        self.P_ion_levels = []  # scalar per level

        for Zi in self.Z_list:
            # ADK rate w(t)
            w_t = np.array([ion_model.rate(Ea, species, Zi) for Ea in self.E_abs])
            self.w_levels.append(w_t)

            # Survival probability S(t) = exp(-∫ w dt)
            W_int = cumulative_trapezoid(w_t, self.t_grid, initial=0.0)
            S_t = np.exp(-W_int)
            self.S_levels.append(S_t)

            # Time-step spacing
            dt_local = np.empty_like(self.t_grid)
            dt_local[:-1] = np.diff(self.t_grid)
            dt_local[-1] = dt_local[-2]

            dP = w_t * S_t * dt_local
            P_ion = dP.sum()

            self.dP_levels.append(dP)
            self.P_ion_levels.append(P_ion)

        # Stack for easier handling: (n_levels, N)
        self.dP_levels = np.vstack(self.dP_levels)  # (nZ, N)
        self.w_levels = np.vstack(self.w_levels)  # (nZ, N)
        self.S_levels = np.vstack(self.S_levels)  # (nZ, N)

        # Total ionization probability (sum of all levels)
        self.P_ion_total = float(self.dP_levels.sum())

    # -----------------------------
    # Helpers to select levels
    # -----------------------------
    def _level_indices(self, levels) -> Sequence[int]:
        """
        Map 'levels' specifier into indices in self.Z_list.
        levels:
          - "all" → all indices
          - int   → that Z only
          - list/tuple of Z → subset
        """
        if levels == "all":
            return list(range(len(self.Z_list)))

        if np.isscalar(levels):
            Zi = int(levels)
            try:
                idx = self.Z_list.index(Zi)
            except ValueError:
                raise ValueError(f"Requested Z={Zi} not in Z_list={self.Z_list}")
            return [idx]

        # assume iterable
        idxs = []
        for Zi in levels:
            Zi = int(Zi)
            try:
                idx = self.Z_list.index(Zi)
            except ValueError:
                raise ValueError(f"Requested Z={Zi} not in Z_list={self.Z_list}")
            idxs.append(idx)
        return idxs

    # -----------------------------
    # Plot interface
    # -----------------------------
    def plot(
        self,
        kind: str = "px",
        levels="all",
        bins: int = 200,
        ax=None,
        cmap="viridis",
        normalize: bool = False,
        label: str = None,
    ):
        """
        Plot the MDF in various projections.

        Parameters
        ----------
        kind : {"px", "py", "pz", "pxpy"}
            - "px": 1D f(p_x)
            - "py": 1D f(p_y)
            - "pz": 1D f(p_z)
            - "pxpy": 2D f(p_x, p_y) as imshow
        levels : "all", int, or sequence of int
            Z levels to include.
        bins : int
            Number of histogram bins.
        ax : matplotlib Axes, optional
            If given, plot on this Ax; otherwise create a new figure.
        cmap : str
            Colormap for 2D plots.
        normalize : bool
            Normalize the selected distribution to sum=1?
        label : str, optional
            Label for the legend (useful when over-plotting multiple calls).
        """

        idxs = self._level_indices(levels)
        selected_Z = [self.Z_list[i] for i in idxs]

        # Combined probability weights
        dP_sel = self.dP_levels[idxs, :].sum(axis=0)
        if normalize and dP_sel.sum() > 0:
            dP_sel = dP_sel / dP_sel.sum()

        # -----------------
        # 1D px/py/pz plots
        # -----------------
        if kind in ("px", "py", "pz"):
            comp = {"px": 0, "py": 1, "pz": 2}[kind]
            p = self.p_grid[:, comp]

            pmin, pmax = p.min(), p.max()
            edges = np.linspace(pmin, pmax, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            F, _ = np.histogram(p, bins=edges, weights=dP_sel)

            # --- Create axes if needed ---
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 4))

            # Choose label automatically if none given
            if label is None:
                label = f"Z={selected_Z}"

            ax.plot(centers, F, label=label)
            ax.set_xlabel(rf"$p_{kind[-1]}\ [m_e c]$")
            ax.set_ylabel("Probability density")
            ax.grid(True)
            ax.legend()

            # Only set title if ax was newly created
            if ax is None:
                ax.set_title(f"MDF in {kind}")

            return ax

        # -----------------
        # 2D px–py plot
        # -----------------
        elif kind == "pxpy":
            px = self.p_grid[:, 0]
            py = self.p_grid[:, 1]

            px_edges = np.linspace(px.min(), px.max(), bins + 1)
            py_edges = np.linspace(py.min(), py.max(), bins + 1)

            H, xedges, yedges = np.histogram2d(px, py, bins=[px_edges, py_edges], weights=dP_sel)

            if normalize and H.sum() > 0:
                H = H / H.sum()

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(H.T, origin="lower", extent=extent, aspect="equal", interpolation="nearest", cmap=cmap)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Probability density")

                title_Z = ", ".join(str(z) for z in selected_Z)
                ax.set_title(f"MDF in $(p_x,p_y)$ for Z={title_Z}")
            else:
                # If an axis is given, overlaying multiple 2D plots isn't useful.
                # Raise an explicit error.
                raise ValueError("Cannot overlay multiple pxpy plots on same axes.")

            ax.set_xlabel(r"$p_x\ [m_e c]$")
            ax.set_ylabel(r"$p_y\ [m_e c]$")

            return ax

        else:
            raise ValueError(f"Unknown kind={kind!r}. Use 'px', 'py', 'pz', or 'pxpy'.")
