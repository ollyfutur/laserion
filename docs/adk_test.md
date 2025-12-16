# ADK Ionization Rate Comparison Test (C vs Python)

This test validates that the **C implementation** of the ADK ionization model produces
the **same ionization rates** as the **Python implementation** in `laserion.ionization`.

The comparison is exhaustive and deterministic:

- **All gases** supported by the ADK tables
- **All available charge states** for each gas
- **Electric field magnitudes**
  $
  |E| = 10^k,\quad k = k_{\min}, \ldots, k_{\max}
  $
- Direct comparison of the **scalar ADK rate** $ w(|E|) $

No random numbers, no integration over time, and no MDF logic are involved.  
This test validates the **atomic ADK kernel** that will later be called inside
grid- and MPI-parallel workflows.

---

## 1. What is being tested

For each tuple: 

$(\text{species}, Z, |E|)$


the test compares:

- Ionization energy $ E_{\mathrm{ion}} $ (in eV)
- ADK ionization rate $ w(|E|) $ (in 1/fs)

computed by:

- **C code** (`csrc/tests/test_adk.c`)
- **Python code** (`laserion.ionization.ADKModel`)

Agreement is checked using:
- strict comparison of ionization energies
- linear tolerance on the rate
- logarithmic tolerance on the rate (robust across many orders of magnitude)

---

## 2. Build the C test executable

From the project root:

```bash
cd csrc
make clean
make
cd ..
```
This must produce the executable:
```bash
csrc/bin/test_adk
```

## 3. Run the comparison
Ensure Python uses the local source tree:
```bash
export PYTHONPATH=$PWD/src
```
Run the test:
```bash
python3 tests/compare_adk.py --Emin-exp 0 --Emax-exp 9 --c-exe csrc/bin/test_adk
```

This sweeps:
  $$
  |E| = 10^k,\quad k = 0, \ldots, 9
  $$


for all gases and all charge states.

## 4. Output modes
### 4.1 Default (recommended)
```bash
python3 tests/compare_adk.py --c-exe csrc/bin/test_adk
```

Output:

- Only a final summary
- Suitable for automated testing / CI

Example:
`
ADK TEST PASSED: all species, all Z, all fields
`

### 4.2 Print only mismatches

```bash 
python3 tests/compare_adk.py --c-exe csrc/bin/test_adk --print-mismatch
```

Prints only rows that fail tolerances: `[RATE] Xe Z=5 E=1.0e+06  C=3.21e-03 PY=3.20e-03 dlog=1.3e-08`

### 4.3 Print all values (debug / inspection)
```bash
python3 tests/compare_adk.py --c-exe csrc/bin/test_adk --print-all
```

Prints one line per $(\text{species}, Z, |E|)$:
```bash
Xe  5  1.0e+06   3.214567e-03   3.214567e-03  1.22e-13
```
Columns:
```bash
species  Z   |E|     w_C           w_PY          |log(w_C) - log(w_PY)|
```

## 5. Tolerances

Default tolerances (can be overridden):

- Ionization energy:
    - absolute tolerance: 1e-12 eV
- Ionization rate:
    - linear: rtol = 1e-12, atol = 1e-15
    - logarithmic: |log(w_C) - log(w_PY)| <= 1e-10

The logarithmic comparison is essential because ADK rates span many orders of magnitude.

## 6. Exit codes
- 0 → all comparisons passed
- 1 → one or more mismatches detected
- 2 → build/runtime/configuration error