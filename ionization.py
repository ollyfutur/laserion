from .base import IonizationModel
from scipy.special import gamma
import numpy as np

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
