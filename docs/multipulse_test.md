# MultiPulse Test (C vs Python): E and A

This test validates that the **C** and **Python** implementations produce the same:
- electric field **E(t, r)**
- vector potential **A(t, r)** (computed by the *same trapezoid integration* rule)

for a **MultiPulse** that is the superposition of **three** pulses:

1. Plane wave transverse + **elliptical** polarization  
2. Gaussian transverse + **linear** polarization  
3. Hermite-Gaussian HG(3,4) transverse + **circular** polarization

Each pulse has a slightly different:
- duration (GaussianTemporal `tau`)
- beam start position `r_start`
- propagation direction `k_vec`
- phase offset `phase_0`

## Files involved

- C generator: `csrc/tests/test_multipulse.c` → builds `csrc/bin/test_multipulse`
- Python comparator: `tests/compare_multipulse.py`

The C executable prints **CSV to stdout**, and prints timing lines to **stderr**.

## Build the C executable

From project root:

```bash
cd csrc
make clean
make
cd ..
```

You should have:

```bash
csrc/bin/test_multipulse
```

If you do not yet compile this target, add it to your `csrc/Makefile` similarly to the other test targets
(e.g. `test_pulse`, `test_temporal`, ...).

## Run the comparison

Example run (10 random points):

```bash
python3 tests/compare_multipulse.py \
  --c-exe csrc/bin/test_multipulse \
  --N 10 --seed 1 \
  --tspan 2000 --xspan 50 --yspan 50 --zspan 50 \
  --Atmin -4000 --Atmax 4000 --Adt 0.05 \
  --E_rtol 1e-12 --E_atol 1e-12 \
  --A_rtol 1e-4 --A_atol 1e-8 \
  --print-frac 0.2
```

### What you will see

- C timing line (from stderr), for the C generator
- Python timing lines for:
  - running the C subprocess
  - evaluating E at N points
  - integrating A at N points
- statistics for component-wise differences:
  - mean
  - RMS
  - max absolute deviation
- PASS/FAIL for E and A

## Notes on numerical tolerance

`A(t, r)` is obtained by numerical integration, so the strictness of `rtol/atol`
should be tuned in proportion to `Adt` and the size of the integration window.

If you reduce `Adt`, the agreement should improve, but both C and Python will take longer.

## Why this test is valuable

This MultiPulse test simultaneously exercises:

- multiple propagation directions (basis construction for polarization)
- multiple transverse profiles (plane / Gaussian / Hermite HG(3,4))
- superposition logic (E_total)
- *consistent* A(t) construction from E_total (trapezoid rule)

