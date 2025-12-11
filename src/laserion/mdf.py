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

    def __init__(self, pulse, species: str, Z, r, ion_model: Optional[IonizationModel] = None, envelope_cut: float = 1e-6, dt: Optional[float] = None):
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
            a default ~ laser period / 300.
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
    def plot(self, kind: str = "px", levels="all", bins: int = 200, normalize: bool = False, label: str = None, figsize=None, **kwargs):
        """
        Plot the MDF in various projections.

        Parameters
        ----------
        kind : {"px", "py", "pz",
                "pxpy", "pxpz", "pypx", "pypz", "pzpx", "pzpy"}
            or sequence of "px","py","pz" for 1D plots.

            1D:
            - "px":  f(p_x)
            - "py":  f(p_y)
            - "pz":  f(p_z)
            If `kind` is a sequence, e.g. ["px","py","pz"], all requested
            1D components are plotted on the same axes (using the SAME
            level selection / group logic described below).

            2D:
            - "pxpy", "pxpz", "pypx", "pypz", "pzpx", "pzpy"
            The first component goes on the horizontal axis, the second on
            the vertical axis. Example:
                "pxpz":  x-axis = p_x, y-axis = p_z
                "pzpx":  x-axis = p_z, y-axis = p_x

            2D kinds must be used alone (not in a list with others).

        levels :
            - "all" → includes all charge states (one group)
            - int or list/tuple of int → that set of Z (one group)
            - list/tuple of groups → multiple groups, each rendered as a
            separate curve in 1D plots. A group can be:
                "all"
                int
                list/tuple of int
            Example:
                levels = ["all", 1, 2]
                → curves for: all Z, Z=1, Z=2

                levels = ["all", [1], [2,3]]
                → curves for: all Z, Z=1, Z=2–3

            For 2D plots, only a single group is allowed (not a list of groups).

        bins : int
            Number of histogram bins.
        normalize : bool
            Normalize each group's distribution to sum=1.
        label : str, optional
            For the simple case: one 1D kind, one level group.
            In multi-group mode, automatic labels are generated unless you
            over-plot manually outside this helper.
        figsize : tuple or None
            Figure size for a new figure (passed to plt.subplots).
        **kwargs :
            Axis kwargs (both 1D & 2D):
                xlim, ylim, xlabel, ylabel, title
            Additional kwargs (2D only):
                passed to imshow (cmap, norm, vmin, vmax, interpolation, ...)
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # --------------------------------------------------
        # 1) Normalize "kind" to a list (for 1D case)
        # --------------------------------------------------
        if isinstance(kind, (list, tuple)):
            kinds = list(kind)
        else:
            kinds = [kind]

        valid_1d = {"px", "py", "pz"}
        valid_2d = {"pxpy", "pxpz", "pypx", "pypz", "pzpx", "pzpy"}
        valid_all = valid_1d | valid_2d

        for k in kinds:
            if k not in valid_all:
                raise ValueError(f"Unknown kind={k!r}. " "Use 'px','py','pz' for 1D or one of " "'pxpy','pxpz','pypx','pypz','pzpx','pzpy' for 2D.")

        if len(kinds) > 1 and any(k in valid_2d for k in kinds):
            raise ValueError("2D (pxpy/pxpz/...) kinds cannot be combined with other kinds.")

        # --------------------------------------------------
        # 2) Parse levels: detect single-group vs multi-group
        # --------------------------------------------------
        def _is_group_spec(obj):
            """Return True if obj looks like a 'group spec' for levels."""
            if isinstance(obj, str):
                return obj == "all"
            if isinstance(obj, (int, np.integer)):
                return True
            if isinstance(obj, (list, tuple)) and all(isinstance(z, (int, np.integer)) for z in obj):
                return True
            return False

        # multi-group if: levels is a sequence AND any element itself looks like a group spec
        multi_group = False
        level_groups = []

        if isinstance(levels, (list, tuple)):
            # If this is something like [1,2,3] (plain ints), treat as single group.
            if all(isinstance(z, (int, np.integer)) for z in levels) and "all" not in levels:
                # Single group with these Z values
                level_groups = [levels]
            else:
                # This is multi-group: e.g. ["all", 1, 2] or ["all",[1],[2,3]]
                # Keep the items as separate group specs
                level_groups = list(levels)
                multi_group = True
        else:
            # Single group (string "all" or int)
            level_groups = [levels]

        # For 2D plots we only allow a single group
        if kinds[0] in valid_2d and len(level_groups) > 1:
            raise ValueError("2D MDF plots (pxpy/pxpz/...) accept only a single 'levels' group.")

        # Build group index lists and default labels per group
        group_indices = []
        group_labels = []

        for g in level_groups:
            idxs = self._level_indices(g)
            group_indices.append(idxs)

            # Build a default legend label for this group
            if isinstance(g, str) and g == "all":
                group_labels.append("all")
            else:
                # Translate idxs → actual Z values
                Zs = [self.Z_list[i] for i in idxs]
                if len(Zs) == 1:
                    group_labels.append(f"Z={Zs[0]}")
                else:
                    group_labels.append("Z=" + ",".join(str(z) for z in Zs))

        # Special case: single group AND a single 1D kind → we allow 'label' override
        single_group_single_kind = len(level_groups) == 1 and len(kinds) == 1 and kinds[0] in valid_1d
        if label is not None and single_group_single_kind:
            group_labels[0] = label

        # --------------------------------------------------
        # 3) Split axis vs image kwargs
        # --------------------------------------------------
        AXIS_KEYS = {"xlim", "ylim", "xlabel", "ylabel", "title"}
        axis_kwargs = {k: v for k, v in kwargs.items() if k in AXIS_KEYS}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in AXIS_KEYS}

        # --------------------------------------------------
        # 4) 2D case (pxpy, pxpz, ...)
        # --------------------------------------------------
        if len(kinds) == 1 and kinds[0] in valid_2d:
            token = kinds[0]
            comp_x = token[:2]  # e.g. "px"
            comp_y = token[2:]  # e.g. "py"

            comp_index = {"px": 0, "py": 1, "pz": 2}
            ix = comp_index[comp_x]
            iy = comp_index[comp_y]

            # Only one group allowed here
            idxs = group_indices[0]
            dP_sel = self.dP_levels[idxs, :].sum(axis=0)
            if normalize and dP_sel.sum() > 0:
                dP_sel = dP_sel / dP_sel.sum()

            vx = self.p_grid[:, ix]
            vy = self.p_grid[:, iy]

            vx_edges = np.linspace(vx.min(), vx.max(), bins + 1)
            vy_edges = np.linspace(vy.min(), vy.max(), bins + 1)

            H, xedges, yedges = np.histogram2d(vx, vy, bins=[vx_edges, vy_edges], weights=dP_sel)

            if normalize and H.sum() > 0:
                H = H / H.sum()

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig, ax = plt.subplots()

            im = ax.imshow(
                H.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                **other_kwargs,
            )

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Probability density")

            xcomp = comp_x[-1]  # 'x','y','z'
            ycomp = comp_y[-1]

            ax.set_xlabel(axis_kwargs.get("xlabel", rf"$p_{xcomp}\ [m_e c]$"))
            ax.set_ylabel(axis_kwargs.get("ylabel", rf"$p_{ycomp}\ [m_e c]$"))

            # Build a title including the Z of this group
            Zs = [self.Z_list[i] for i in group_indices[0]]
            title_Z = ", ".join(str(z) for z in Zs)
            x0, y0, z0 = self.r
            default_title = f"MDF at (x={x0}, y={y0}, z={z0}) for Z={title_Z}"
            ax.set_title(axis_kwargs.get("title", default_title))

            if "xlim" in axis_kwargs:
                ax.set_xlim(axis_kwargs["xlim"])
            if "ylim" in axis_kwargs:
                ax.set_ylim(axis_kwargs["ylim"])

            plt.show()
            return ax

        # --------------------------------------------------
        # 5) 1D case: px / py / pz (possibly several kinds)
        # --------------------------------------------------
        comp_index = {"px": 0, "py": 1, "pz": 2}

        # Collect full p-ranges over all requested components
        p_components = {}
        for k in kinds:
            if k not in valid_1d:
                raise ValueError(f"1D plotting supports only {valid_1d}, but got {k!r}.")
            idx = comp_index[k]
            p_components[k] = self.p_grid[:, idx]

        pmin = min(p.min() for p in p_components.values())
        pmax = max(p.max() for p in p_components.values())
        edges = np.linspace(pmin, pmax, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        # For each group and each kind, plot one curve
        for g_idx, idxs in enumerate(group_indices):
            dP_g = self.dP_levels[idxs, :].sum(axis=0)
            if normalize and dP_g.sum() > 0:
                dP_g = dP_g / dP_g.sum()

            for k in kinds:
                p = p_components[k]
                F, _ = np.histogram(p, bins=edges, weights=dP_g)

                if len(level_groups) == 1 and len(kinds) > 1:
                    # Single level group, multiple kinds → label by kind
                    curve_label = k
                elif len(level_groups) > 1 and len(kinds) == 1:
                    # Multiple level groups, single kind → label by group
                    curve_label = group_labels[g_idx]
                elif len(level_groups) == 1 and len(kinds) == 1:
                    # Single group, single kind: we already handled possible override
                    curve_label = group_labels[g_idx]
                else:
                    # Multiple groups & multiple kinds: label by both
                    curve_label = f"{group_labels[g_idx]} ({k})"

                ax.plot(centers, F, label=curve_label)

        # Axis labels
        if len(kinds) == 1:
            comp = kinds[0][-1]  # 'x','y','z'
            xlabel_default = rf"$p_{comp}\ [m_e c]$"
        else:
            xlabel_default = r"$p\ [m_e c]$"

        ax.set_xlabel(axis_kwargs.get("xlabel", xlabel_default))
        ax.set_ylabel(axis_kwargs.get("ylabel", "Probability density"))
        ax.grid(True)
        ax.legend()

        if "xlim" in axis_kwargs:
            ax.set_xlim(axis_kwargs["xlim"])
        if "ylim" in axis_kwargs:
            ax.set_ylim(axis_kwargs["ylim"])

        if "title" in axis_kwargs:
            ax.set_title(axis_kwargs["title"])
        else:
            x0, y0, z0 = self.r
            joined_kinds = "/".join(kinds)
            default_title = f"{joined_kinds} at (x={x0}, y={y0}, z={z0})"

            ax.set_title(axis_kwargs.get("title", default_title))

        plt.show()
        return ax
