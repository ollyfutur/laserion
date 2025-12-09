import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import Optional, Sequence
import matplotlib.pyplot as plt
from .ionization import IonizationModel, ADKModel


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
