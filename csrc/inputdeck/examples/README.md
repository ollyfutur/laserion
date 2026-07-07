# Helium ionization examples

Two small examples showing how to set up and run `laserion` for a Helium gas
ionized by a tightly focused laser pulse, and how to look at the output.

Both use a **1D grid along the propagation axis `z`**, sampled on-axis
(`x = y = 0`). The code still evaluates the full 3D Gaussian-beam fields — on
axis the amplitude scales as `w0 / w(z)`, so it peaks at the focus (`z = zf`)
and falls off before and after it. With a small waist `w0` the Rayleigh length
`z0 = π w0² / λ` is short (~16 µm here), so the intensity — and therefore the
ionization — varies strongly along `z`.

Helium is used because it has only two levels (He → He⁺ → He²⁺), which keeps
the picture simple.

## Requirements

- `laserion` built (`make` in `csrc/`, binary at `csrc/bin/laserion`)
- Python with `numpy`, `h5py`, `matplotlib` (and Jupyter, or the VS Code
  Jupyter extension) for the plots

## Example 1 — [`He_focus/`](He_focus/): a single focused pulse

Records the laser field at three positions (**before focus / at focus / after
focus**) and the ionization fraction along `z`.

```bash
cd He_focus
laserion                 # or:  mpirun -n 4 laserion   (z is split across ranks)
```

Then open [`plots.ipynb`](plots.ipynb) and run the cells. In the figure:

- **Field vs time:** the pulse is weak before the focus, strong at the focus,
  weak after — the beam focusing. Each trace is centred at a different time
  (`t ≈ z/c`, the propagation delay).
- **Ionization vs z:** He⁺ is fully ionized in the focal region and drops to
  zero away from it; He²⁺ appears only near the focus.

## Example 2 — [`He_intensity/`](He_intensity/): scanning the intensity

Three runs at increasing peak field `E0`, same focused-Helium setup, showing
three ionization regimes:

| run    | `E0` (GV/m) | result                                          |
|--------|-------------|-------------------------------------------------|
| `low`  | 200         | first level fully ionized at focus; second negligible |
| `mid`  | 300         | second level **partially** ionized near the focus     |
| `high` | 1200        | both levels fully ionized across the domain           |

```bash
for d in He_intensity/low He_intensity/mid He_intensity/high; do
    ( cd "$d" && laserion )
done
```

Then run the Example 2 cell in [`plots.ipynb`](plots.ipynb) to see the three
ionization profiles side by side.

## Output layout

Each run writes an `MS/` directory:

```
MS/cache/        Ex,Ey,Ez,Ax,Ay,Az on the (t, z) grid
MS/field/        laser field vs t at the requested z positions   (Example 1)
MS/ioniz_frac/   ion_frac_Z00 = mean charge <Z>;  Z01, Z02 = fraction reaching each charge state
```

The `ion_frac_Z0N` files hold the **cumulative** fraction reaching at least
charge state `N` (so `Z01` is the singly-ionized fraction, `Z02` the
doubly-ionized fraction). Every HDF5 file has one named dataset plus an `AXIS`
group whose `AXIS1`/`AXIS2` give the `[min, max]` of each axis;
[`plots.ipynb`](plots.ipynb) shows how to open them in Python.

The `MS/` directories are regenerated on every run; delete them with
`rm -rf He_focus/MS He_intensity/*/MS` to reclaim the space.

## Plotting

[`plots.ipynb`](plots.ipynb) opens the HDF5 data with `h5py` and plots the
fields and ionization fraction. Open it in VS Code or Jupyter and run the cells
— the loading helpers are small and easy to adapt for your own plots.
