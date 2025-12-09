from .base import TransverseProfile
import numpy as np
from typing import Optional

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
