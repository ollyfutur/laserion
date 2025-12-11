import numpy as np
from .base import LaserPulse, TransverseProfile, TemporalProfile
from .polarization import Polarization
from .temporal_profile import GaussianTemporal
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from typing import Optional, Sequence

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

        # --- default dt: laser period / 300 ---
        if dt is None:
            T_laser = self.wavelength / self.c
            dt = T_laser / 300.0

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

    def plot(self, component, plane: str, *, t: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, xrange=(-5.0, 5.0), yrange=(-5.0, 5.0), N: int = 400, r_for_A=None, figsize=None, **kwargs):
        """
        Plot electric or vector potential field components.

        Parameters
        ----------
        component : str or sequence of str
            One or several of "Ex","Ey","Ez","Ax","Ay","Az".
            For 1D planes, multiple components are overlaid on the same axes.
            For 2D planes, exactly one component must be given.
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
        figsize : tuple or None
            Figure size passed to plt.subplots / plt.figure.
            If None, matplotlib rcParams["figure.figsize"] is used.
        **kwargs :
            Extra keyword arguments:
            - for 1D: applied to the line plots (color, linestyle, ...) and to
                    the Axes via keys xlim, ylim, xlabel, ylabel, title
            - for 2D: split into axis kwargs (xlim, ylim, xlabel, ylabel, title)
                    and image kwargs passed to imshow (cmap, norm, etc.).
        """

        # Normalize component to a list
        if isinstance(component, str):
            components = [component]
        else:
            components = list(component)

        valid_components = {"Ex", "Ey", "Ez", "Ax", "Ay", "Az"}
        for c in components:
            if c not in valid_components:
                raise ValueError(f"Unknown component {c!r}. Must be one of {sorted(valid_components)}.")

        # Helper: axis kwargs (for Axes methods) vs others
        AXIS_KEYS = {"xlim", "ylim", "xlabel", "ylabel", "title"}

        def split_kwargs(all_kwargs):
            axis_kwargs = {k: v for k, v in all_kwargs.items() if k in AXIS_KEYS}
            other_kwargs = {k: v for k, v in all_kwargs.items() if k not in AXIS_KEYS}
            return axis_kwargs, other_kwargs

        r0 = np.array([x, y, z], dtype=float)
        axis_map = {"x": 0, "y": 1, "z": 2}

        # =================================================
        #            1D plots: "t", "x", "y", "z"
        # =================================================
        if plane in ("t", "x", "y", "z"):
            var = np.linspace(*xrange, N)
            axis_kwargs, line_kwargs = split_kwargs(kwargs)

            # figure and axis
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            if plane == "t":
                xlabel_default = "t [fs]"
            else:
                xlabel_default = f"{plane} [µm]"

            # Loop over each requested component and plot sequentially
            for comp in components:
                if comp.startswith("E"):
                    idx_map = {"Ex": 0, "Ey": 1, "Ez": 2}
                    idx = idx_map[comp]

                    if plane == "t":
                        # vary time, position fixed
                        vals = np.zeros_like(var)
                        for i, tt in enumerate(var):
                            vals[i] = self.E(tt, r0)[idx]
                    else:
                        # vary one spatial coordinate, keep t fixed
                        ax_idx = axis_map[plane]
                        vals = np.zeros_like(var)
                        for i, coord in enumerate(var):
                            r_vec = r0.copy()
                            r_vec[ax_idx] = coord
                            vals[i] = self.E(t, r_vec)[idx]

                elif comp.startswith("A"):
                    idx_map = {"Ax": 0, "Ay": 1, "Az": 2}
                    idx = idx_map[comp]

                    rr_cache = np.array([x, y, z]) if r_for_A is None else np.asarray(r_for_A)
                    # Build cache (no-op if already built for this r)
                    self.build_A_cache(rr_cache)

                    if plane == "t":
                        vals = np.zeros_like(var)
                        for i, tt in enumerate(var):
                            vals[i] = self.A(tt, rr_cache)[idx]
                    else:
                        ax_idx = axis_map[plane]
                        vals = np.zeros_like(var)
                        for i, coord in enumerate(var):
                            r_vec = rr_cache.copy()
                            r_vec[ax_idx] = coord
                            vals[i] = self.A(t, r_vec)[idx]

                else:
                    raise ValueError("component must be one of Ex,Ey,Ez,Ax,Ay,Az")

                # Plot this component
                ax.plot(var, vals, label=comp, **line_kwargs)

            # Labels, grid, title, limits
            ax.set_xlabel(axis_kwargs.get("xlabel", xlabel_default))
            # If only one component, default ylabel is that component; else generic
            if len(components) == 1:
                default_ylabel = components[0]
            else:
                default_ylabel = ""
            ax.set_ylabel(axis_kwargs.get("ylabel", default_ylabel))
            ax.grid(True)

            if len(components) > 1:
                ax.legend()

            if "xlim" in axis_kwargs:
                ax.set_xlim(axis_kwargs["xlim"])
            if "ylim" in axis_kwargs:
                ax.set_ylim(axis_kwargs["ylim"])

            if "title" in axis_kwargs:
                ax.set_title(axis_kwargs["title"])
            else:
                if plane == "t":
                    # fields vs time at fixed (x,y,z)
                    title_str = f"(x={x}, y={y}, z={z})"
                else:
                    # fields vs spatial coordinate at fixed time
                    # e.g. Ex vs x at (y=0,z=0) and t = 120 fs
                    other_coords = {"x": (y, z), "y": (x, z), "z": (x, y)}
                    # pick the two remaining coordinates
                    if plane == "x":
                        title_str = f"(y={y}, z={z}, t={t})"
                    elif plane == "y":
                        title_str = f"(x={x}, z={z}, t={t})"
                    else:  # plane == "z"
                        title_str = f"(x={x}, y={y}, t={t})"
            ax.set_title(title_str)
            plt.show()
            return ax

        # =================================================
        #      2D MAPS: "xy", "yx", "xz", "zx", "yz", "zy"
        # =================================================
        if len(plane) != 2 or any(ax_c not in "xyz" for ax_c in plane):
            raise ValueError("plane must be 't','x','y','z' or any 2-letter combo of 'x','y','z' " "(e.g. 'xy','xz','yz','zx','yx','zy').")

        if len(components) != 1:
            raise ValueError("2D plots currently support a single component only.")

        comp = components[0]
        axis_kwargs, img_kwargs = split_kwargs(kwargs)

        # Determine which field to sample
        if comp.startswith("E"):
            idx_map = {"Ex": 0, "Ey": 1, "Ez": 2}
            idx = idx_map[comp]
            field_fn = lambda tt, rr: self.E(tt, rr)
        elif comp.startswith("A"):
            idx_map = {"Ax": 0, "Ay": 1, "Az": 2}
            idx = idx_map[comp]
            rr_cache = np.array([x, y, z]) if r_for_A is None else np.asarray(r_for_A)
            self.build_A_cache(rr_cache)
            field_fn = lambda tt, rr: self.A(tt, rr)
        else:
            raise ValueError("component must be one of Ex,Ey,Ez,Ax,Ay,Az")

        ax1 = axis_map[plane[0]]  # horizontal
        ax2 = axis_map[plane[1]]  # vertical

        X = np.linspace(*xrange, N)
        Y = np.linspace(*yrange, N)
        XX, YY = np.meshgrid(X, Y)

        vals = np.zeros_like(XX)
        for i in range(N):
            for j in range(N):
                r_vec = r0.copy()
                r_vec[ax1] = XX[i, j]
                r_vec[ax2] = YY[i, j]
                vals[i, j] = field_fn(t, r_vec)[idx]

        if figsize is not None:
            fig, ax2d = plt.subplots(figsize=figsize)
        else:
            fig, ax2d = plt.subplots()

        im = ax2d.imshow(
            vals,
            extent=(xrange[0], xrange[1], yrange[0], yrange[1]),
            origin="lower",
            aspect="auto",
            **img_kwargs,
        )
        plt.colorbar(im, label=comp)

        ax2d.set_xlabel(axis_kwargs.get("xlabel", plane[0] + " [µm]"))
        ax2d.set_ylabel(axis_kwargs.get("ylabel", plane[1] + " [µm]"))

        if "xlim" in axis_kwargs:
            ax2d.set_xlim(axis_kwargs["xlim"])
        if "ylim" in axis_kwargs:
            ax2d.set_ylim(axis_kwargs["ylim"])

        if "title" in axis_kwargs:
            ax2d.set_title(axis_kwargs["title"])
        else:
            ax2d.set_title(f"{comp} in {plane}-plane at t={t}")

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

    def build_A_cache(self, r, t_min=None, t_max=None, dt=None, envelope_cut=1e-8):
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

        # --- default dt: use smallest wavelength among pulses, period/300 ---
        if dt is None:
            lambdas = []
            for p in self.pulses:
                if hasattr(p, "wavelength"):
                    lambdas.append(p.wavelength)
            if len(lambdas) == 0:
                raise ValueError("dt must be provided if sub-pulses lack 'wavelength' attribute.")
            lambda_min = min(lambdas)
            T_min = lambda_min / self.c
            dt = T_min / 300.0

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
    def plot(self, component, plane: str, *, t: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, xrange=(-5.0, 5.0), yrange=(-5.0, 5.0), N: int = 400, r_for_A=None, show_total: bool = True, show_pulses: bool = False, labels: Optional[Sequence[str]] = None, figsize=None, **kwargs):
        """
        Plot electric or vector potential field components for a MultiPulse.

        Parameters
        ----------
        component : str or sequence of str
            One or several of "Ex","Ey","Ez","Ax","Ay","Az".
            For 1D planes, multiple components are overlaid on the same axes.
            For 2D planes, exactly one component must be given.
            All components in a single call must be of the same type (all E* or all A*).
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
            If True (1D only), also plot individual sub-pulse contributions.
        labels : sequence of str, optional
            Labels for each sub-pulse when show_pulses=True.
            If None, labels like "pulse 0", "pulse 1", ... are used.
        figsize : tuple or None
            Figure size passed to plt.subplots / plt.figure.
            If None, matplotlib rcParams["figure.figsize"] is used.
        **kwargs :
            Extra keyword arguments:
            - for 1D: applied to the line plots (color, linestyle, ...) and to
                    the Axes via keys xlim, ylim, xlabel, ylabel, title
            - for 2D: split into axis kwargs (xlim, ylim, xlabel, ylabel, title)
                    and image kwargs passed to imshow (cmap, norm, etc.).
        """

        # Normalize component(s) to a list
        if isinstance(component, str):
            components = [component]
        else:
            components = list(component)

        valid_components = {"Ex", "Ey", "Ez", "Ax", "Ay", "Az"}
        for c in components:
            if c not in valid_components:
                raise ValueError(f"Unknown component {c!r}. Must be one of {sorted(valid_components)}.")

        # Ensure all components are either all E* or all A*
        prefixes = {c[0] for c in components}
        if len(prefixes) != 1:
            raise ValueError("All components in a single MultiPulse.plot call must be of the same type " "(all starting with 'E' or all starting with 'A').")
        prefix = prefixes.pop()  # 'E' or 'A'

        # ---------- Prepare total and per-pulse field functions ----------
        r0 = np.array([x, y, z], dtype=float)
        axis_map = {"x": 0, "y": 1, "z": 2}

        if prefix == "E":
            # total E
            def total_field_fn(tt, rr):
                return self.E(tt, rr)

            # per-pulse E
            pulse_field_fns = [lambda tt, rr, p=p: p.E(tt, rr) for p in self.pulses]

        else:  # prefix == "A"
            rr_cache = np.array([x, y, z]) if r_for_A is None else np.asarray(r_for_A)
            # Build cache for total A
            self.build_A_cache(rr_cache)

            def total_field_fn(tt, rr):
                # A uses cached r; rr is ignored
                return self.A(tt, rr_cache)

            # per-pulse A: ensure each has cache at the same rr_cache
            pulse_field_fns = []
            for p in self.pulses:
                if hasattr(p, "build_A_cache"):
                    p.build_A_cache(rr_cache)
                    pulse_field_fns.append(lambda tt, rr, pp=p: pp.A(tt, rr_cache))
                else:
                    pulse_field_fns.append(lambda tt, rr: np.zeros(3))

        # Split kwargs: axis vs others
        AXIS_KEYS = {"xlim", "ylim", "xlabel", "ylabel", "title"}

        def split_kwargs(all_kwargs):
            axis_kwargs = {k: v for k, v in all_kwargs.items() if k in AXIS_KEYS}
            other_kwargs = {k: v for k, v in all_kwargs.items() if k not in AXIS_KEYS}
            return axis_kwargs, other_kwargs

        # =================================================
        #            1D plots: "t", "x", "y", "z"
        # =================================================
        if plane in ("t", "x", "y", "z"):
            var = np.linspace(*xrange, N)
            axis_kwargs, line_kwargs = split_kwargs(kwargs)

            # figure and axis
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            if plane == "t":
                xlabel_default = "t [fs]"
            else:
                xlabel_default = f"{plane} [µm]"

            # sanity check for labels if show_pulses
            if show_pulses:
                if labels is not None and len(labels) != len(self.pulses):
                    raise ValueError("labels length must match number of pulses.")

            # Loop over each requested component and plot sequentially
            for comp in components:
                if comp.startswith("E"):
                    idx_map = {"Ex": 0, "Ey": 1, "Ez": 2}
                else:
                    idx_map = {"Ax": 0, "Ay": 1, "Az": 2}
                idx = idx_map[comp]

                # --- total field for this component ---
                vals_total = np.zeros_like(var)
                if plane == "t":
                    # vary time, position fixed
                    for i, tt in enumerate(var):
                        vals_total[i] = total_field_fn(tt, r0)[idx]
                else:
                    # vary one spatial coordinate, keep t fixed
                    ax_idx = axis_map[plane]
                    for i, coord in enumerate(var):
                        r_vec = r0.copy()
                        r_vec[ax_idx] = coord
                        vals_total[i] = total_field_fn(t, r_vec)[idx]

                if show_total:
                    ax.plot(var, vals_total, label=f"{comp} (total)", **line_kwargs)

                # --- per-pulse contributions for this component ---
                if show_pulses:
                    for k, f_pulse in enumerate(pulse_field_fns):
                        vals_pulse = np.zeros_like(var)
                        if plane == "t":
                            for i, tt in enumerate(var):
                                vals_pulse[i] = f_pulse(tt, r0)[idx]
                        else:
                            ax_idx = axis_map[plane]
                            for i, coord in enumerate(var):
                                r_vec = r0.copy()
                                r_vec[ax_idx] = coord
                                vals_pulse[i] = f_pulse(t, r_vec)[idx]

                        base_label = labels[k] if labels is not None else f"pulse {k}"
                        lab = f"{base_label} ({comp})"
                        ax.plot(var, vals_pulse, "--", label=lab, **line_kwargs)

            # Labels, grid, limits, title
            ax.set_xlabel(axis_kwargs.get("xlabel", xlabel_default))

            if len(components) == 1:
                default_ylabel = components[0]
            else:
                default_ylabel = ""
            ax.set_ylabel(axis_kwargs.get("ylabel", default_ylabel))

            ax.grid(True)

            if show_total or show_pulses or len(components) > 1:
                ax.legend()

            if "xlim" in axis_kwargs:
                ax.set_xlim(axis_kwargs["xlim"])
            if "ylim" in axis_kwargs:
                ax.set_ylim(axis_kwargs["ylim"])

            if "title" in axis_kwargs:
                ax.set_title(axis_kwargs["title"])
            else:
                if plane == "t":
                    # fields vs time at fixed (x,y,z)
                    title_str = f"(x={x}, y={y}, z={z})"
                else:
                    # fields vs spatial coordinate at fixed time
                    if plane == "x":
                        title_str = f"(y={y}, z={z}, t={t})"
                    elif plane == "y":
                        title_str = f"(x={x}, z={z}, t={t})"
                    else:  # plane == "z"
                        title_str = f"(x={x}, y={y}, t={t})"
                ax.set_title(title_str)

            plt.show()
            return ax

        # =================================================
        #      2D MAPS: "xy", "yx", "xz", "zx", "yz", "zy"
        # =================================================
        if len(plane) != 2 or any(ax_c not in "xyz" for ax_c in plane):
            raise ValueError("plane must be 't','x','y','z' or any 2-letter combo of 'x','y','z' " "(e.g. 'xy','xz','yz','zx','yx','zy').")

        if len(components) != 1:
            raise ValueError("2D plots currently support a single component only.")

        if show_pulses:
            raise ValueError("show_pulses=True is not supported for 2D plots.")

        comp2d = components[0]
        axis_kwargs, im_kwargs = split_kwargs(kwargs)

        # index selection for this component
        if comp2d.startswith("E"):
            idx_map = {"Ex": 0, "Ey": 1, "Ez": 2}
            idx = idx_map[comp2d]

            def field_fn_2d(tt, rr):
                return self.E(tt, rr)

        else:  # A*
            idx_map = {"Ax": 0, "Ay": 1, "Az": 2}
            idx = idx_map[comp2d]

            rr_cache = np.array([x, y, z]) if r_for_A is None else np.asarray(r_for_A)
            self.build_A_cache(rr_cache)

            def field_fn_2d(tt, rr):
                # A uses cached r; rr is ignored
                return self.A(tt, rr_cache)

        axis_map = {"x": 0, "y": 1, "z": 2}
        ax1 = axis_map[plane[0]]  # horizontal
        ax2 = axis_map[plane[1]]  # vertical

        X = np.linspace(*xrange, N)
        Y = np.linspace(*yrange, N)
        XX, YY = np.meshgrid(X, Y)

        vals = np.zeros_like(XX)
        for i in range(N):
            for j in range(N):
                r_vec = r0.copy()
                r_vec[ax1] = XX[i, j]
                r_vec[ax2] = YY[i, j]
                vals[i, j] = field_fn_2d(t, r_vec)[idx]

        if figsize is not None:
            fig, ax2d = plt.subplots(figsize=figsize)
        else:
            fig, ax2d = plt.subplots()

        im = ax2d.imshow(
            vals,
            extent=(xrange[0], xrange[1], yrange[0], yrange[1]),
            origin="lower",
            aspect="auto",
            **im_kwargs,
        )
        plt.colorbar(im, ax=ax2d, label=comp2d)

        ax2d.set_xlabel(axis_kwargs.get("xlabel", plane[0] + " [µm]"))
        ax2d.set_ylabel(axis_kwargs.get("ylabel", plane[1] + " [µm]"))

        if "xlim" in axis_kwargs:
            ax2d.set_xlim(axis_kwargs["xlim"])
        if "ylim" in axis_kwargs:
            ax2d.set_ylim(axis_kwargs["ylim"])

        if "title" in axis_kwargs:
            ax2d.set_title(axis_kwargs["title"])
        else:
            ax2d.set_title(f"{comp2d} in {plane}-plane at t={t}")

        plt.show()
        return ax2d
