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
> - Even if `[field_cache].out_dir` is set, `run.c` overwrites it at runtime to `MS/cache`.

---

## `[run]`

Controls general runtime context and some global physics choices.

### `working_dir` (string)

- **Default:** `"."`
- If not `"."`, `run.c` will:
  1. create the directory (like `mkdir -p`)
  2. `chdir()` into it
- All outputs are then produced relative to this directory.

### `gas` (string)

- **Default:** `"H"`
- Selects the gas species for ionization/particle related routines.
- Must be supported by `ionization_model.c` tables (validated via `ionization_species_supported()`).
- Used by:
  - ionization fraction diagnostics
  - particle diagnostics (to build `Z_list` up to `Zmax`)

### `ionization_model` (string)

- **Default:** `"adk"`
- **Currently only `"adk"` is accepted.**
- Any other value errors out during parsing.

---

## `[field_cache]`

Controls the cached field generation used by downstream diagnostics.

> Important runtime behavior:
>
> - `run.c` always uses cache directory `MS/cache` (inside working_dir).
> - It **overwrites** `sim.field_cache.out_dir` to `"MS/cache"`.

### `mode` (string)

- **Default:** `"auto"`
- Allowed values (case-insensitive, with some aliases):
  - `"auto"`
  - `"compute"`

Runtime meaning in `run.c`:

- `auto`:
  - If `MS/cache` exists and is “complete” (Ex/Ey/Ez and Ax/Ay/Az), check compatibility:
    - if compatible, reuse it
    - if incompatible, recompute cache
  - If missing, compute cache.
- `compute`:
  - Forces recomputation of the cache.

---

## `[grid]` (REQUIRED)

Defines the sampling domain for cache generation and diagnostics.

### `t_min`, `t_max` (double)

- **Defaults:** `t_min=-200.0`, `t_max=200.0`
- Time range (in units of fs).
- Must satisfy `t_max > t_min`.

### `dt` (double)

- **Default:** `0.02`
- Time step for sampling the cached fields.
- Must satisfy `dt > 0`.

### `spatial_axes` (string)

- **Default:** `"z"` (because defaults set `ax1=Z`, and `has_ax2=false`)
- Required key in `[grid]` (parser errors if missing).
- Allowed values: 1 or 2 distinct characters from `{x, y, z}`.
  - Examples: `"z"`, `"x"`, `"zx"`, `"xz"`, `"xy"`, `"yz"`.
- Meaning:
  - `ax1` is the primary spatial axis
  - optional `ax2` is the secondary spatial axis
  - If only one axis is given, the domain is effectively 1D in space.
  - If two axes are given, the domain is 2D in space.
- Any axis not present in `spatial_axes` is treated as **fixed** (see `fixed` below).

#### `fixed.x`, `fixed.y`, `fixed.z` (double)

- **Defaults:** `0.0` for each.
- Used when an axis is not part of the spatial domain.
- Also used as defaults for particle diagnostics if that axis is not in the domain.

Example:

```toml
[grid]
spatial_axes = "zx"
fixed.y = 0.0
```

### `ax1_min`, `ax1_max`, `dx1` (double)

- **Defaults:** `ax1_min=-5.0`, `ax1_max=5.0`, `dx1=0.1`
- Defines the sampling interval along `ax1`.
- Requirements:
  - `ax1_max > ax1_min`
  - `dx1 > 0` (indirectly enforced by the derived-grid routine)
- Internally, `ax1_n` is derived as:

  `ax1_n = round((ax1_max - ax1_min) / dx1) + 1`

  (includes endpoints)

### `ax2_min`, `ax2_max`, `dx2` (double) (only if `spatial_axes` has 2 axes)

- **Defaults:** `ax2_min=-5.0`, `ax2_max=5.0`, `dx2=0.1`
- Same meaning as for `ax1_*`, but for the second spatial axis.
- `ax2_n` is derived similarly.

---

## `[[laser]]` (REQUIRED array)

Defines one or more lasers. At least one `[[laser]]` table is required.

### `type` (string) (optional)

- **Allowed values:** `"standard"`
- Anything else errors out.
- This has only one option available now but it may be useful to distinguish other kind of laser initialization later

### `E0` (double)

- **Default:** `100.0`
- Peak field amplitude in GV/m.
- Must satisfy `E0 >= 0`.

### `wavelength` (double)

- **Default:** `0.8`
- Laser wavelength in µm.
- Must satisfy `wavelength > 0`.

### `phase0` (double)

- **Default:** `0.0`
- Initial carrier phase.

### `k_vec` (array[3] of double)

- **Default:** `[0.0, 0.0, 1.0]`
- Propagation direction vector (not necessarily normalized here, but usually interpreted as direction).

### `r_start` (array[3] of double)

- **Default:** `[0.0, 0.0, 0.0]`
- Starting point / reference position for the pulse.

### `use_retarded_time` (bool)

- **Default:** `true`
- Enables using retarded-time evaluation for the pulse (as implemented in the laser evaluation).

### `temporal_type` (string)

- **Default:** `"gaussian"`
- **Allowed:** `"gaussian"` only (currently).
- If omitted, defaults apply.

### `tau` (double)

- **Default:** `30.0`
- Temporal duration parameter for the temporal profile in units of fs.
- Must satisfy `tau > 0`.

### `transverse_type` (string)

- **Default:** `"gaussian"`
- Allowed values:
  - `"gaussian"`
  - `"hermite"`

### `w0` (double)

- **Default:** `4.0`
- Transverse beam waist parameter in units of µm.
- Must satisfy `w0 > 0`.

### `zf` (double)

- **Default:** `0.0`)
- Focus position along propagation axis in units of µm.

### `herm_lm` (array[2] of int) (required if `transverse_type="hermite"`)

- Example: `herm_lm = [1, 0]`
- Sets Hermite mode indices `(l, m)`.
- If `transverse_type="hermite"` and this is missing: parsing errors out.

### `polarization` (string)

- **Default:** `"linear"`
- Allowed values:
  - `"linear"`
  - `"circular"`
  - `"jones"`

### `angle` (double)

- **Default:** `0.0`
- For linear polarization: polarization angle.
- For other types: may still be read but interpretation depends on implementation.

### `sense` (string) (required for circular polarization)

- Allowed values:
  - `"right"`
  - `"left"`
- If `polarization="circular"` and `sense` is absent: parsing errors out.

### Jones parameters: `p1`, `p2`, `delta` (double)

- **Defaults:** `p1=1.0`, `p2=0.0`, `delta=0.0`
- If any of these is provided, the deck marks `has_jones=true`.
- If `polarization="jones"`, then at least one of these must be provided; otherwise parsing errors out.
- **Jones parameters (`p1`, `p2`, `delta`)** define the complex polarization state via Jones calculus.  
  `p1` and `p2` set the relative amplitudes of two orthogonal field components, and `delta` is their relative phase (in degrees).

  Reference: https://en.wikipedia.org/wiki/Jones_calculus


---

## `[[field_diag]]`

Requests field diagnostics computed **from the cache** and written to `MS/field/`.

Each element is a table in the top-level array `field_diag`.

### `axes` (string) (required)

- Must be 1 or 2 distinct characters from `{t, x, y, z}`.
- Examples:
  - `"t"` : 1D diagnostic sweeping time only
  - `"z"` : 1D diagnostic sweeping z only
  - `"tz"`: 2D diagnostic sweeping time and z
  - `"xy"`: 2D diagnostic sweeping x and y
- Any axis not included in `axes` uses the fixed positions in `pos_*`.

### `components` (array of strings) (required)

- Array length: 1 to 8
- Each entry must be one of: `Ex`, `Ey`, `Ez`, `Ax`, `Ay`, `Az` (case-insensitive in parsing).
- Example:
  - `components = ["Ex", "Ez"]`

### `pos_x`, `pos_y`, `pos_z`, `pos_t` (double, optional)

- **Defaults:** `0.0`
- Fixed coordinate values used for dimensions that are **not** swept by `axes`.
- Example:
  - If `axes="tz"`, then `pos_x` and `pos_y` are the fixed transverse positions.

---

## `[ionization_frac]`

Enables ionization fraction diagnostics computed from cache.

### Presence-based enable

- If the table `[ionization_frac]` exists, then `enabled=true`.
- There are currently **no keys** inside the table; it is effectively a boolean switch.

Runtime behavior:
- Computes ionization fraction from `MS/cache` and writes to `MS/ioniz_frac`.

---

## `[[phase_space]]` (MDF phase-space diagnostics)

Requests MDF phase-space diagnostics computed from cache and written to `MS/mdf/`.

Each element is a table in the top-level array `phase_space`.

### `phase_space` (string) (required)

Selects phase-space projection kind.

Allowed values:

- 1D:
  - `"px"`, `"py"`, `"pz"`
- 2D ordered pairs (order matters):
  - `"pxpy"`, `"pypx"`
  - `"pxpz"`, `"pzpx"`
  - `"pypz"`, `"pzpy"`

Interpretation:
- 1D kinds produce a 1D histogram over the chosen momentum component.
- 2D kinds produce a 2D histogram over ordered components `(p1, p2)`.

### `bins1` (table) (required)

Defines the binning for the first momentum axis.

Keys:
- `nbins` (int, >0)
- `min` (double)
- `max` (double, must satisfy `max > min`)

### `bins2` (table)

- Required for 2D kinds.
- Forbidden for 1D kinds (if present in 1D: parsing errors out).
- Same structure as `bins1`.

### `envelope_cut` (double, optional)

- **Default:** `0.0`
- Threshold parameter passed to MDF routines (used to cut low-envelope regions, depending on implementation).

### `normalize_sum_to_1` (bool, optional)

- **Default:** `false`
- If true, normalizes the MDF histogram so that its total sum equals 1.

### `region.xmin`, `region.xmax`, `...` (double, optional)

Spatial clipping region for the diagnostic.

- If absent: diagnostic uses the full cached spatial domain.
- If present: any missing bounds default to the cache bounds.

Possible keys:
- `xmin`, `xmax`
- `ymin`, `ymax`
- `zmin`, `zmax`

Validation:
- Only axes present in the grid domain are validated.
- If both bounds for an axis are finite, then `max > min` is required.

---

## `[[particles]]` (particle diagnostics)

Requests particle diagnostics computed from cache and written to `MS/particles/`.

Each element is a table in the top-level array `particles`.

### `ppc` (int, optional)

- **Default:** `50`
- Particles-per-cell used for sampling/generation.
- Must be `> 0`.

### `seed` (int, optional)

- **Default:** `0`
- Random seed for particle sampling.
- Must be `>= 0` (stored as unsigned long long).

### `region.xmin`, `region.xmax`, `...` (double, optional)

Spatial clipping region for the diagnostic.

Possible keys:
- `xmin`, `xmax`
- `ymin`, `ymax`
- `zmin`, `zmax`

Behavior:
- If an axis is NOT in the grid domain, it is forced to the corresponding fixed coordinate.
- If bounds are finite, must satisfy `max > min`.
- If region table is absent:
  - domain axes default to “full domain”
  - non-domain axes are fixed to `[grid].fixed.*`

### `sampling.nx`, `sampling.ny`, `sampling.nz` (double, optional)

Overrides the number of spatial sample points used for particle diagnostics.

Behavior:
- Only axes present in the grid domain are meaningful.
- For axes not in the domain, runtime forces the corresponding `n` to 1.

Default sampling behavior (when `sampling` is absent):
- Uses the grid resolution (`ax1_n`, `ax2_n`) on the active axes as best as possible.
- Uses 1 along non-active axes.

---

## Minimal example template

```toml
[run]
working_dir = "."
gas         = "He"

[field_cache]
mode = "auto"

[grid]
t_min         = -200.0
t_max         =  200.0
dt            =  0.02
spatial_axes  = "z"
fixed.x       = 0.0
fixed.y       = 0.0 
ax1_min       = -10.0
ax1_max       =  10.0
dx1           =  0.05

[[laser]]
E0                = 100.0
wavelength        = 1.0
phase0            = 0.0
k_vec             = [0.0, 0.0, 1.0]
r_start           = [0.0, 0.0, 0.0]
use_retarded_time = true
temporal_type     = "gaussian"
tau               = 30.0
transverse_type   = "gaussian"
w0                = 4.0
zf                = 0.0
polarization      = "linear"
angle             = 0.0

[[laser]]
E0                = 20.0
wavelength        = 0.8
phase0            = 0.0
k_vec             = [0.0, 0.0, 1.0]
r_start           = [0.0, 0.0, 0.0]
use_retarded_time = true
temporal_type     = "gaussian"
tau               = 25.0
transverse_type   = "gaussian"
w0                = 12.0
zf                = 0.0
polarization      = "circular"
angle             = 0.0
sense             = "left"

[[field_diag]]
axes       = "tz"
components = ["Ex","Ez"]

[[field_diag]]
axes       = "z"
components = ["Ex","Ez"]
pos_t      = 1.0

[ionization_frac]

[[phase_space]]
phase_space = "pxpy"
bins1.nbins = 200
bins1.min   = -1.0
bins1.max   =  1.0
bins2.nbins = 200
bins2.min   = -1.0
bins2.max   =  1.0
region.zmin = -2.0
region.zmax =  2.0


[[particles]]
ppc = 50
seed = 1234
sampling.nz = 256
```
