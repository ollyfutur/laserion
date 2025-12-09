from abc import ABC, abstractmethod
import numpy as np

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
