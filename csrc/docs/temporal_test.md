# Temporal Envelope Validation (C vs Python)

This document explains how to validate that the **C implementation** of the
Gaussian temporal envelope matches exactly the **Python implementation**
used in `laserion`.

The comparison is numerical and optionally visual.

---

## 1. Build the C test executable

From the project root directory:

```bash
cd csrc
make clean
make
cd ..
```

This builds the executable:

```text
csrc/bin/test_temporal
```

---

## 2. Run the numerical comparison

Ensure that Python uses the **local** `laserion` sources (not an installed version):

```bash
export PYTHONPATH=$PWD/src
```

Run the comparison script:

```bash
python3 tests/compare_temporal.py \
  --tau 60 \
  --tmin -200 \
  --tmax 200 \
  --dt 0.5 \
  --c-exe csrc/bin/test_temporal
```

The script compares the two implementations point-by-point and reports:

- maximum absolute difference
- maximum relative difference

The test is considered successful if both values are below the configured
tolerances (`--atol`, `--rtol`).

---

## 3. Plot the comparison (optional)

To visually confirm agreement between implementations:

```bash
python3 tests/plot_temporal.py \
  --tau 60 \
  --tmin -200 \
  --tmax 200 \
  --dt 0.5 \
  --c-exe csrc/bin/test_temporal
```

This produces a plot showing:
- Python result (continuous line)
- C result (discrete markers)

They should overlap within numerical precision.

---

## Notes

- The Python reference uses the existing `GaussianTemporal` class from `laserion`.
- The C code is treated as a black box and queried only through the executable.
- No output files are written unless explicitly enabled in the scripts.

