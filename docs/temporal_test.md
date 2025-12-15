# Temporal Envelope Test (C vs Python)

This test validates that the **C implementation** of the Gaussian temporal envelope
matches exactly the **Python implementation** in `laserion`.

The workflow is:
1. The C executable computes the envelope from CLI parameters and writes CSV to stdout.
2. A Python script calls the C executable, recomputes the same envelope using the
   Python `laserion` code, compares both, and optionally plots them.
3. No CSV files are created or left behind.

## Build the C test executable

From the project root:

```bash
cd csrc
make clean
make
cd ..
```

This produces the executable:

```bash
csrc/bin/test_temporal
```

## Run the comparison

Ensure Python imports the local `laserion` sources:

```bash
export PYTHONPATH=$PWD/src
```

Run:

```bash
python3 tests/compare_temporal.py --tau 60 --tmin -200 --tmax 200 --dt 0.5 --c-exe csrc/bin/test_temporal`
```

The script reports:
- number of points
- maximum absolute difference
- maximum relative difference
- whether the comparison passed

Exit code is 0 on success, 1 on failure.

## Plot C vs Python (optional)

```bash
python3 tests/compare_temporal.py --tau 60 --tmin -200 --tmax 200 --dt 0.5 --c-exe csrc/bin/test_temporal --plot
```

A figure appears with:
- solid line: Python result
- dashed line: C result

## Notes

- All numerical parameters are provided via the command line.
- The C code writes CSV to stdout only; Python parses it directly.
- No temporary files are created.

