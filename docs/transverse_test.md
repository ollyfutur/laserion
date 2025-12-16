# Transverse Profile Comparison Test

This test verifies that the **C implementation** of the transverse laser envelope and phase
matches the **Python implementation** in `laserion.transverse_profile`.

The comparison is done point-by-point at randomly chosen positions in the **beam frame**
and reports both **per-point differences** and **summary statistics**.

---

## 1. Build the C test executable

From the `laserion/csrc` directory:

```bash
make clean
make
```

This produces the executable:

```
laserion/csrc/bin/test_transverse
```

---

## 2. What the test does

1. The **C code**:
   - Randomly samples `N` points `(x', y', z')` around the laser focus.
   - Computes:
     - transverse envelope
     - transverse phase
   - Prints results as CSV **to stdout**.

2. The **Python script**:
   - Calls the C executable.
   - Recomputes envelope and phase using the **Python `laserion` code**.
   - Prints:
     - per-point differences (C − Python)
     - mean / RMS / max error statistics
   - Exits with non-zero status if tolerances are violated.

No CSV files are kept; everything is handled in memory.

---

## 3. Run the comparison (Gaussian example)

From the project root (`laserion/`):

```bash
export PYTHONPATH=$PWD/src

python3 tests/compare_transverse.py \
  --profile gaussian \
  --wavelength 10 \
  --w0 5 \
  --zf 0 \
  --N 2000 \
  --seed 1 \
  --print-all \
  --c-exe csrc/bin/test_transverse
```

---

## 4. Hermite–Gaussian example

```bash
python3 tests/compare_transverse.py \
  --profile hermite \
  --wavelength 10 \
  --w0 5 \
  --zf 0 \
  --l 2 --m 1 \
  --N 2000 \
  --seed 1 \
  --print-all \
  --c-exe csrc/bin/test_transverse
```

---

## 5. Sampling region control

By default, the sampled region is chosen automatically from the beam parameters.
You can override it explicitly:

```bash
python3 tests/compare_transverse.py \
  --profile gaussian \
  --wavelength 10 \
  --w0 5 --zf 0 \
  --xspan 8 \
  --yspan 8 \
  --zspan 200 \
  --N 2000 \
  --seed 1 \
  --print-all \
  --c-exe csrc/bin/test_transverse
```

All coordinates are in **µm** and are **beam-frame coordinates** `(x', y', z')`.

---

## 6. Transverse Test with Non-Zero Focus Position

To test agreement away from the focus, set a non-zero focal position `zf`.

Example: focus located at z′ = +150 µm.

```bash
python3 tests/compare_transverse.py \
  --profile gaussian \
  --wavelength 10 \
  --w0 5 \
  --zf 150 \
  --zspan 400 \
  --xspan 8 \
  --yspan 8 \
  --N 2000 \
  --seed 2 \
  --print-all \
  --c-exe csrc/bin/test_transverse
```

---

## 7. Output interpretation

- **Per-point table**:
  - Shows `(x, y, z)`, C values, Python values, and differences.
- **Summary statistics**:
  - Mean, RMS, and max absolute error for:
    - envelope
    - wrapped phase difference
- **Exit status**:
  - `0` → C and Python agree within tolerances
  - `!= 0` → mismatch detected

