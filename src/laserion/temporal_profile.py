from .base import TemporalProfile
import numpy as np

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
