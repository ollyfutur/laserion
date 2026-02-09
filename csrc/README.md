# laserion inputdeck (TOML) — Options Reference

This document describes the inputdeck options currently **parsed by `inputdeck.c`** and **used by `run.c`**.

## High-level structure

A valid input file is a TOML document with (at minimum):

- `[grid]` (required)
- `[[laser]]` (required, at least one element)

Everything else is optional:

- `[run]`
- `[field_cache]`
- `[[field_diag]]`
- `[ionization_frac]`
- `[[phase_space]]`
- `[[particles]]`

> Notes on output layout:
>
> - `run.c` forces the output structure into a folder `MS/` inside `[run].working_dir`:
>   - `MS/cache/`      (field cache files)
>   - `MS/field/`      (field diagnostics)
>   - `MS/ioniz_frac/` (ionization fraction diagnostics)
>   - `MS/mdf/`        (phase-space MDF diagnostics)
>   - `MS/particles/`  (particle diagnostics)

---

## `[run]`
Controls general runtime context and some global physics choices.

### `working_dir` (string)
Outputs are then produced relative to this directory (**Default:** `"."`, *i.e.*, the current directory)

### `gas` (string)
**Options available:** `"H"`, `"He"`, `"Li"`, `"N"`, `"O"`, `"Ar"`, `"Ne"`, `"Xe"`. **Default:** `"H"`.
    
---

## `[field_cache]`

### `mode` (string)
**Options available:** `"auto"`, `"compute"` (**Default:** `"auto"`)
- `auto`:
  - If `MS/cache` exists and is “complete” (Ex/Ey/Ez and Ax/Ay/Az), check compatibility:
    - if compatible, reuse it
    - if incompatible, recompute cache
  - If missing, compute cache.
- `compute`:
  - Forces recomputation of the cache.

---
## `[grid]`
Defines the sampling domain for cache generation and diagnostics.

### `t_min`, `t_max`, `dt` (double)
Time range and time step in units of fs.

### `spatial_axes` (string)
Allowed values: 1 or 2 distinct characters from `{x, y, z}` (**default:** `"z"`).

### `fixed.x`, `fixed.y`, `fixed.z` (double)
**Defaults:** `0.0` for each. Used when an axis is not part of the spatial domain.
Example:
```toml
[grid]
spatial_axes = "zx"
fixed.y = 0.0
```
### `ax1_min`, `ax1_max`, `dx1` (double)
Defines the spatial domain and grid interval along `ax1` in units of μm.

### `ax2_min`, `ax2_max`, `dx2` (double) (only if `spatial_axes` has 2 axes)
Defines the spatial domain and grid interval along `ax2` in units of μm.

---

## `[[laser]]`

Defines one or more lasers. At least one `[[laser]]` table is required.

### `E0` (double)
Peak field amplitude in GV/m (**default** `100`)

### `wavelength` (double)
Laser wavelength in µm  (**default** `0.8`).

### `phase0` (double)
Initial carrier phase in degrees (**default** `0.0`).

### `k_vec` (array[3] of double) and `r_start` (array[3] of double)
- `k_vec` defines the propagation direction vector (**default** `k_vec = [0.0, 0.0, 1.0]`, *i.e.*, propagation along `z`). It is normalized internally such that $\hat k = \mathbf k_{vec}/|\mathbf k_{vec}|$
- `r_start` defines the reference position of the pulse in the lab frame, corresponding to the centre (peak) of the temporal envelope at $t=0$ (**default** `[0.0, 0.0, 0.0]`).
- Together, `k_vec` and `r_start` uniquely define the propagation (beam) axis: $\mathbf r_{prop} (s) = \mathbf{r}_{start} + s\, \hat{k}$
### `temporal_type` (string)
Temporal profile (only `"gaussian"` is available for now).
### `tau` (double)
Temporal duration parameter for the temporal profile in units of fs (**default:** `30.0`).

### `transverse_type` (string)
Allowed values: `"plane_wave"`, `"gaussian"`, `"hermite"`  (**default:** `"plane_wave"`)

### `w0` (double)
Transverse beam waist parameter in units of µm (**default:** `4.0`). Used in the case of `transverse_type` `"gaussian"` and `"hermite"`.

### `zf` (double)
Focus position along propagation axis (see `k_vec` and `r_start` above) in units of µm (**default:** `0.0`). Used in the case of `transverse_type` `"gaussian"` and `"hermite"`.

### `herm_lm` (array[2] of int) (required if `transverse_type="hermite"`)
Sets Hermite mode indices `(l, m)` if `transverse_type` is `"hermite"` (**default:** `herm_lm = [0, 0]`)

### `polarization` (string)
Allowed values: `"linear"`, `"circular"`, `"jones"` (**default:** `"linear"`)

### `angle` (double)
For linear polarization: polarization angle in degrees (**default:** `0.0`). For other types: may still be read but interpretation depends on implementation.

### `sense` (string)
Required for circular polarization. Allowed values: `"right"` and `"left"`.

### Jones parameters: `p1`, `p2`, `delta` (double)
**Jones parameters (`p1`, `p2`, `delta`)** define the complex polarization state via Jones calculus. `p1` and `p2` set the relative amplitudes of two orthogonal field components, and `delta` is their relative phase in degrees. Reference: https://en.wikipedia.org/wiki/Jones_calculus


---

## `[[field_diag]]`

Requests field diagnostics computed **from the cache** and written to `MS/field/`.

Each element is a table in the top-level array `field_diag`.

### `axes` (string)

Must be 1 or 2 distinct characters from `{t, x, y, z}`. Example: `axes = "xt"` Any axis not included in `axes` uses the fixed positions in `pos_*`.

### `components` (array of strings) (required)

Array in which each entry must be one of: `Ex`, `Ey`, `Ez`, `Ax`, `Ay`, `Az`. Example: `components = ["Ex", "Ez"]`.

### `pos_x`, `pos_y`, `pos_z`, `pos_t` (double, optional)

Fixed coordinate values used for dimensions that are **not** swept by `axes` (**defaults:** `0.0`). Example: If `axes="tz"`, then `pos_x` and `pos_y` are the fixed transverse positions.

---

## `[ionization_frac]`

Enables ionization fraction diagnostics computed from cache.

### Presence-based enable

- If the table `[ionization_frac]` exists, then `enabled=true`.

Runtime behavior:
- Computes ionization fraction from `MS/cache` and writes to `MS/ioniz_frac`.

---

## `[[phase_space]]` (MDF phase-space diagnostics)

Requests MDF phase-space diagnostics computed from cache and written to `MS/mdf/`.

Each element is a table in the top-level array `phase_space`.

### `phase_space` (string)

Selects phase-space projection kind.

Allowed values: 1D - `"px"`, `"py"`, `"pz"`; 2D ordered pairs (order matters): - `"pxpy"`, `"pypx"`, `"pxpz"`, `"..."`

### `bins1.nbins`, `bins1.min`, `bins1.max`
Defines the binning for the first momentum axis.

### `bins2.nbins`, `bins2.min`, `bins2.max`
Defines the binning for the second momentum axis.

### `region.xmin`, `region.xmax`, `region.ymin`, `...`
Spatial clipping region for the diagnostic. - If absent: diagnostic uses the full cached spatial domain.

---

## `[[particles]]` (particle diagnostics)

Requests particle diagnostics computed from cache and written to `MS/particles/`.

### `ppc` (int, optional)
Particles-per-cell per ionization level used for sampling/generation (**default:** `50`).

### `seed` (int, optional)
Random seed for particle sampling (**default:** `0`).

### `region.xmin`, `region.xmax`, `region.ymin`, `...`
Spatial clipping region for the diagnostic. - If absent: diagnostic uses the full cached spatial domain.

### `sampling.nx`, `sampling.ny`, `sampling.nz`
Overrides the number of spatial sample points used for particle diagnostics.

---

## Minimal example template

```toml
[run]
working_dir = "."
gas         = "He"

[field_cache]
mode = "auto"

[grid]
t_min        = -200.0
t_max        =  200.0
dt           =  0.02
spatial_axes = "z"
fixed.x      = 0.0
fixed.y      = 0.0
ax1_min      = -10.0
ax1_max      =  10.0
dx1          =  0.05

[[laser]]
E0              = 100.0
wavelength      = 1.0
phase0          = 0.0
k_vec           = [0.0, 0.0, 1.0]
r_start         = [0.0, 0.0, 0.0]
temporal_type   = "gaussian"
tau             = 30.0
transverse_type = "gaussian"
w0              = 4.0
zf              = 0.0
polarization    = "linear"
angle           = 0.0

[[laser]]
E0              = 100.0
wavelength      = 0.8
phase0          = 0.0
k_vec           = [0.0, 0.0, 1.0]
r_start         = [0.0, 0.0, 0.0]
temporal_type   = "gaussian"
tau             = 25.0
transverse_type = "gaussian"
w0              = 12.0
zf              = 0.0
polarization    = "circular"
angle           = 0.0
sense           = "left"

[[field_diag]]
axes       = "tz"
components = ["Ex", "Ey"]

[[field_diag]]
axes       = "t"
components = ["Ex", "Ey"]
pos_z      = 0

[ionization_frac]

[[phase_space]]
phase_space = "pxpy"
bins1.nbins =  500
bins1.min   = -0.065
bins1.max   =  0.065
bins2.nbins =  500
bins2.min   = -0.065
bins2.max   =  0.065
region.zmin = -2.0
region.zmax =  2.0


[[particles]]
ppc         = 50
seed        = 1234
sampling.nz = 256
```
