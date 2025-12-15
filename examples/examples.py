# %% Examples: Laser fields, polarization states, and MDF diagnostics
#
# This file is meant as a collection of "drop-in" usage examples.
# Each block defines a pulse (or multi-pulse), produces a few field plots,
# and (optionally) computes/plots the MDF at a fixed spatial point.
#
# Conventions used by this codebase (as implemented in core/mdf):
#   - E_0 in GV/m
#   - wavelength in µm
#   - time in fs
#   - position r = (x,y,z) in µm
#   - MDF momenta in units of m_e c
#
# Notes on plotting:
#   - SinglePulse.plot / MultiPulse.plot:
#       component: "Ex","Ey","Ez" or "Ax","Ay","Az" (or list in 1D)
#       plane: "t","x","y","z" (1D) or combinations like "zx", "zy", ...
#       xrange/yrange set the displayed coordinate ranges for that plane.
#   - MDF.plot:
#       kind: "px","py","pz" (1D) or "pxpy","pxpz","pypz", etc. (2D)
#       levels: "all" or explicit Z subsets/groups
#       For 2D plots, imshow kwargs such as cmap/extent/aspect can be passed.

import numpy as np
from laserion.core import SinglePulse, MultiPulse
from laserion.temporal_profile import GaussianTemporal
from laserion.transverse_profile import PlaneWaveProfile, GaussianTransverse
from laserion.polarization import LinearPolarization, CircularPolarization, Polarization
from laserion.mdf import MDF
from matplotlib.colors import LogNorm

# %% ------------------------------------------------------------------------
# Example 1: Plane-wave + linear polarization (k along +z)
# ---------------------------------------------------------------------------
#
# Purpose:
#   - Basic sanity check of E-field time trace and (x,z) map.
#   - MDF for helium with two charge states, evaluated at the origin.
#
# Physics setup:
#   - Plane wave: no transverse dependence (constant amplitude in x,y)
#   - Linear polarization: "x-like" transverse direction consistent with k_vec
#   - Retarded time enabled: envelope is evaluated at t_eff = t - z'/c (if used internally)

E0 = 150.0  # Peak field [GV/m]
lambda0 = 10  # Wavelength [µm]
tau = 1000.0  # RMS pulse duration [fs]

temporal_1 = GaussianTemporal(tau=tau)  # Gaussian temporal envelope
transverse_1 = PlaneWaveProfile()  # Plane-wave transverse profile
k_vec_1 = (0.0, 0.0, 1.0)  # Propagation direction (lab frame) along +z
polarization_1 = LinearPolarization(
    k_vec=k_vec_1,
    angle=0.0,  # angle=0 picks the "x-like" transverse basis direction
)

# Pulse start position at t=0 (lab frame). For a plane wave, this mainly shifts the phase/arrival.
r_start_1 = (0.0, 0.0, 0.0)

pulse_1 = SinglePulse(
    E_0=E0,
    wavelength=lambda0,
    temporal=temporal_1,
    transverse=transverse_1,
    polarization=polarization_1,
    phase_0=0.0,
    k_vec=k_vec_1,
    use_retarded_time=True,
    r_start=r_start_1,
)

# (a) 1D: Ex(t) at the origin (x=y=z=0)
t_window = 2.0 * tau
_ = pulse_1.plot(
    component="Ex",
    plane="t",
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-t_window, t_window),
    N=10000,
)

# (b) 2D: Ex(z,x) map at t = 0 fs
# plane="zx" means: horizontal axis is z, vertical axis is x.
_ = pulse_1.plot(
    component="Ex",
    plane="zx",
    t=0.0,
    x=0.0,
    y=0.0,
    z=0.0,  # base point; ranges below define the actual scan
    xrange=(-500.0, 500.0),  # z-range [µm]
    yrange=(-20.0, 20.0),  # x-range [µm]
    N=1000,
    cmap="seismic",
)

# (c) 2D: Ex(z,x) map at t = 1000 fs (late-time snapshot)
_ = pulse_1.plot(
    component="Ex",
    plane="zx",
    t=1000.0,
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-500.0, 500.0),
    yrange=(-20.0, 20.0),
    N=1000,
    cmap="seismic",
)

# (d) MDF: helium ionization (Z=1 and Z=2), evaluated at the origin
# dt controls the time-step used to build the cached A(t) grid inside MDF.
mdf = MDF(pulse_1, species="He", Z=[1, 2], r=[0, 0, 0], dt=0.005)

# 1D MDF: plot px for multiple level groups (all, only Z=1, only Z=2)
mdf.plot(kind=["px"], levels=["all", [1], [2]])


# %% ------------------------------------------------------------------------
# Example 2: Plane-wave + circular polarization with oblique propagation
# ---------------------------------------------------------------------------
#
# Purpose:
#   - Validate that for oblique k_vec you still get a consistent transverse polarization.
#   - Compare E(t) and A(t) components.
#   - MDF in 1D and 2D for hydrogen (single level).

E0 = 150.0
lambda0 = 10.0
tau = 300.0

temporal_2 = GaussianTemporal(tau=tau)
transverse_2 = PlaneWaveProfile()

# Oblique propagation direction; the code will normalize it internally where needed.
k_vec_2 = (0.0, 2.0, 1.0)

polarization_2 = CircularPolarization(
    k_vec=k_vec_2,
    angle=0.0,  # sets the orientation of the transverse basis used for circular state
)

r_start_2 = (0.0, 0.0, 0.0)

pulse_2 = SinglePulse(
    E_0=E0,
    wavelength=lambda0,
    temporal=temporal_2,
    transverse=transverse_2,
    polarization=polarization_2,
    phase_0=0.0,
    k_vec=k_vec_2,
    use_retarded_time=True,
    r_start=r_start_2,
)

# Time traces at the origin
t_window = 2.0 * tau

# (a) E-field components vs time
_ = pulse_2.plot(
    component=["Ex", "Ey", "Ez"],
    plane="t",
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-t_window, t_window),
    N=10000,
    lw=2,
)

# (b) Vector potential components vs time
_ = pulse_2.plot(
    component=["Ax", "Ay", "Az"],
    plane="t",
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-t_window, t_window),
    N=10000,
    lw=2,
)

# (c) 2D field map in the (z,y) plane at t = 0
_ = pulse_2.plot(
    component=["Ex"],
    plane="zy",
    x=0.0,
    t=0.0,
    xrange=(-200, 200),  # z-range [µm]
    yrange=(-200, 200),  # y-range [µm]
    N=1000,
    cmap="seismic",
)

# (d) MDF for hydrogen (single ionization level)
mdf = MDF(pulse_2, species="H", Z=[1], r=[0, 0, 0], dt=0.005)

# 1D MDF: plot all components on the same axis
mdf.plot(kind=["px", "py", "pz"], levels=["all"], xlim=(-0.15, 0.15), bins=1000)

# 2D MDF: px vs pz
mdf.plot(kind="pxpz", bins=200, cmap="Blues")


# %% ------------------------------------------------------------------------
# Example 3: MultiPulse = superposition of pulse_1 and pulse_2
# ---------------------------------------------------------------------------
#
# Purpose:
#   - Demonstrate field maps for a superposition.
#   - Compute MDF at an off-axis / displaced evaluation point.

pulse_mp = MultiPulse([pulse_1, pulse_2])

# Field maps for the superposition in the (z,y) plane at x=0
_ = pulse_mp.plot(
    component=["Ex"],
    plane="zy",
    x=0.0,
    t=0.0,
    xrange=(-500, 500),
    yrange=(-500, 500),
    N=1000,
    cmap="seismic",
)

_ = pulse_mp.plot(
    component=["Ex"],
    plane="zy",
    x=0.0,
    t=1000.0,
    xrange=(-500, 500),
    yrange=(-500, 500),
    N=1000,
    cmap="seismic",
)

_ = pulse_mp.plot(
    component=["Ey"],
    plane="zy",
    x=0.0,
    t=0.0,
    xrange=(-200, 200),
    yrange=(-200, 200),
    N=1000,
    cmap="seismic",
)

# MDF evaluation point: r = (x, y, z) in µm
mdf = MDF(pulse_mp, species="He", Z=[1, 2], r=[0, 200, 300], dt=0.005)

# 1D MDF examples
mdf.plot(kind=["px"], levels=["all", [1], [2]])
mdf.plot(kind=["pz"], levels=["all"], bins=1000)

# Field time trace at the same MDF evaluation point
pulse_mp.plot("Ex", "t", x=0, y=200, z=300, xrange=[-1000, 3000], N=10000)

# 2D MDF example: px vs py
# (Note: kind should be a single 2D token; passing a list is usually unnecessary.)
mdf.plot(kind=["pxpy"], levels=["all"], bins=1000, norm=LogNorm(vmin=1e-10, vmax=1e-1), cmap="Blues")


# %% ------------------------------------------------------------------------
# Example 4: Focused Gaussian transverse profile + circular polarization
# ---------------------------------------------------------------------------
#
# Purpose:
#   - Show how GaussianTransverse produces finite transverse structure.
#   - Field maps at early/late times.
#   - MDF in 1D and 2D.

E0 = 250.0
lambda0 = 10
tau = 750.0

temporal_3 = GaussianTemporal(tau=tau)
transverse_3 = GaussianTransverse(w0=200, zf=0)  # waist w0 [µm], focus location zf [µm]
k_vec_3 = (0.0, 0.0, 1.0)

polarization_3 = CircularPolarization(k_vec=k_vec_3, angle=0.0)

r_start_3 = (0.0, 0.0, 0.0)

pulse_3 = SinglePulse(
    E_0=E0,
    wavelength=lambda0,
    temporal=temporal_3,
    transverse=transverse_3,
    polarization=polarization_3,
    phase_0=0.0,
    k_vec=k_vec_3,
    use_retarded_time=True,
    r_start=r_start_3,
)

# Ex(t) at the origin
t_window = 2.0 * tau
_ = pulse_3.plot(component="Ex", plane="t", x=0.0, y=0.0, z=0.0, xrange=(-t_window, t_window), N=10000)

# Ex(z,x) map at t = 0
_ = pulse_3.plot(
    component="Ex",
    plane="zx",
    t=0.0,
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-500.0, 500.0),
    yrange=(-1000.0, 1000.0),
    N=500,
    cmap="seismic",
    vmin=-250,
    vmax=250,
)

# Ex(z,x) map at t = 1000 fs
_ = pulse_3.plot(
    component="Ex",
    plane="zx",
    t=100000.0,
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(pulse_3.c * 100000 - 500.0, pulse_3.c * 100000 + 500.0),
    yrange=(-1000.0, 1000.0),
    N=500,
    cmap="seismic",
    vmin=-250,
    vmax=250,
)

# MDF after the laser passes
mdf = MDF(pulse_3, species="He", Z=[1, 2], r=[0, 0, 0], dt=0.005)

mdf.plot(kind=["px"], levels=["all", [1], [2]], xlim=(-1, 1))
mdf.plot(kind=["pxpy"], levels=["all"], bins=400)


# %% -------------------------------------------------------------------------------------------
# Example 5: Same as Example 4 but with a different carrier phase (phase_0) and shorter duration
# ----------------------------------------------------------------------------------------------
#
# Purpose:
#   - Show how changing phase_0 shifts the carrier oscillation relative to the envelope.
#   - Compare MDF dependence (if any) at a displaced z evaluation point.

E0 = 250.0
lambda0 = 10
tau = 100.0

temporal_4 = GaussianTemporal(tau=tau)
transverse_4 = GaussianTransverse(w0=200, zf=0)
k_vec_4 = (0.0, 0.0, 1.0)

polarization_4 = CircularPolarization(k_vec=k_vec_4, angle=0.0)
r_start_4 = (0.0, 0.0, 0.0)

# phase_0 here is provided in the same units expected by SinglePulse (per your implementation).
pulse_4 = SinglePulse(
    E_0=E0,
    wavelength=lambda0,
    temporal=temporal_4,
    transverse=transverse_4,
    polarization=polarization_4,
    phase_0=90.0,
    k_vec=k_vec_4,
    use_retarded_time=True,
    r_start=r_start_4,
)

t_window = 2.0 * tau
_ = pulse_4.plot(component="Ex", plane="t", x=0.0, y=0.0, z=0.0, xrange=(-t_window, t_window), N=10000)

_ = pulse_4.plot(
    component="Ex",
    plane="zx",
    t=0.0,
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-100.0, 100.0),
    yrange=(-500.0, 500.0),
    N=500,
    cmap="seismic",
    vmin=-250,
    vmax=250,
)


mdf = MDF(pulse_4, species="He", Z=[1, 2], r=[0, 0, 50], dt=0.005)

mdf.plot(kind=["px"], levels=["all", [1], [2]], xlim=(-1, 1))
mdf.plot(kind=["pxpy"], levels=["all"], bins=400)

# Diagnostic: per-level total ionization probabilities
print("fraction of ionization for levels [1, 2] = ", mdf.P_ion_levels)

mdf.plot(kind=["px"], levels=["all", [1], [2]], xlim=(-1, 1))


# %% ------------------------------------------------------------------------
# Example 6: Elliptical polarization using the general Polarization class
# ---------------------------------------------------------------------------
#
# Purpose:
#   - Demonstrate true elliptical polarization via (p1, p2, delta) in a fixed basis.
#   - Validate Ex(t) and Ey(t) are phase-shifted and have different amplitudes.
#   - Demonstrate MDF in 1D and 2D, including imshow kwargs (extent/aspect).

E0 = 150.0
lambda0 = 10
tau = 1000.0

temporal_5 = GaussianTemporal(tau=tau)
transverse_5 = GaussianTransverse(w0=200, zf=0)
k_vec_5 = (0.0, 0.0, 1.0)
r_start_5 = (0.0, 0.0, 0.0)

# Elliptical polarization in the lab x/y basis:
#   p1=1.0 along x (e1), p2=0.5 along y (e2), delta=pi/2 gives quadrature phase shift.
polarization_5 = Polarization(
    e1=(1, 0, 0),
    e2=(0, 1, 0),
    p1=1.0,
    p2=0.5,
    delta=np.pi / 2,
)

pulse_5 = SinglePulse(
    E_0=E0,
    wavelength=lambda0,
    temporal=temporal_5,
    transverse=transverse_5,
    polarization=polarization_5,
    phase_0=0.0,
    k_vec=k_vec_5,
    use_retarded_time=True,
    r_start=r_start_5,
)

# Ex(t), Ey(t) time traces at the origin: should show quadrature and unequal amplitudes
t_window = 2.0 * tau
_ = pulse_5.plot(component=["Ex", "Ey"], plane="t", z=0.0, y=0.0, x=0.0, xrange=(-t_window, t_window), N=10000)

# Field maps (Ex and Ey) in the (z,x) plane at t=0
_ = pulse_5.plot(
    component="Ex",
    plane="zx",
    t=0.0,
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-500.0, 500.0),
    yrange=(-500.0, 500.0),
    N=500,
    cmap="seismic",
    vmin=-150,
    vmax=150,
)

_ = pulse_5.plot(
    component="Ey",
    plane="zx",
    t=0.0,
    x=0.0,
    y=0.0,
    z=0.0,
    xrange=(-500.0, 500.0),
    yrange=(-500.0, 500.0),
    N=500,
    cmap="seismic",
    vmin=-150,
    vmax=150,
)

# MDF at a displaced evaluation point (z=50 µm)
mdf = MDF(pulse_5, species="He", Z=[1, 2], r=[0, 0, 50], dt=0.005)

# 1D MDF: px and py on the same axis (all levels combined)
mdf.plot(kind=["px", "py"], levels=["all"])

# 2D MDF: px vs py with explicit imshow parameters.
# extent/aspect are passed to imshow (useful to enforce a fixed view window or equal aspect ratio).
mdf.plot(
    kind=["pxpy"],
    levels=["all"],
    bins=400,
    cmap="Reds",
    figsize=(10, 6),
    extent=[-0.4, 0.4, -0.4, 0.4],
    aspect="auto",
)

# %%
