# Vector Potential (A) Comparison Test — C vs Python

This document describes how to validate the **vector potential**
$\mathbf{A}(t,\mathbf{r}) $ computed by the C implementation against the
Python implementation in **laserion**.

Unlike the electric field $\mathbf{E} $, the vector potential is obtained by
**numerical time integration**, and therefore **looser numerical tolerances are
both expected and correct**.

---

## 1. Purpose of the test

The goal is to ensure that:

- The **same physical definition** of $\mathbf{A} $ is used in C and Python
- The **same numerical integration scheme** (trapezoidal rule) is applied
- Polarization, transverse profiles, and oblique propagation are handled consistently
- Differences are **purely numerical**, not algorithmic

This test does **not** attempt to validate the absolute convergence of the
integration scheme, only **consistency between implementations**.

---

## 2. Definition being tested

Both C and Python compute:

$$
\mathbf{A}(t,\mathbf{r}) = -\int_{t_\mathrm{min}}^{t}
\mathbf{E}(t',\mathbf{r})\,\mathrm{d}t'
$$

using:

- A **uniform time grid**
- The **trapezoidal rule**
- **Linear interpolation** to evaluate $\mathbf{A} $ at arbitrary times

The integration window $[t_\mathrm{min}, t_\mathrm{max}]$ and time step
$\Delta t $ **must be identical** in C and Python.

---

## 3. Files involved

### C side

- `csrc/tests/test_pulse_A.c`  
  Generates random sample points, computes
  $\mathbf{A}(t,\mathbf{r}) $, and prints CSV rows to stdout.

### Python side

- `tests/compare_pulse_A.py`  
  Runs the C executable, recomputes $\mathbf{A} $ using the same trapezoidal
  rule, and compares results.

---

## 4. Build the C test executable

From the project root:

```bash
cd csrc
make clean
make
cd ..
```

This must produce:

```bash
csrc/bin/test_pulse_A
```

---

## 5. Running the test

Ensure Python uses the local source tree:

```bash
export PYTHONPATH=$PWD/src
```

### 5.1 Standard multi-case test

```bash
python3 tests/compare_pulse_A.py \
  --c-exe csrc/bin/test_pulse_A \
  --Npos 40 --Nt 100 --seed 1 \
  --Atmin -4000 --Atmax 4000 --Adt 0.01
```

---

## 6. Fast single-point sanity check

Because $\mathbf{A} $ requires time integration, runtime scales as:

$$
\mathcal{O}\left( \frac{t_\mathrm{max}-t_\mathrm{min}}{\Delta t} \right)
$$

For a quick check at a **single spacetime point**, use:

```bash
python3 tests/compare_pulse_A.py \
  --c-exe csrc/bin/test_pulse_A \
  --Npos 1 --Nt 1 --seed 1 \
  --Atmin -3000 --Atmax 3000 --Adt 0.1 \
  --print-frac 1.0
```

---

## 7. Printing sampled values

To print a fraction of sampled rows:

```bash
--print-frac 0.01
```

To print **all** sampled rows (use with care):

```bash
--print-frac 1.0
```

---

## 8. Numerical tolerances (IMPORTANT)

### 8.1 Why tight tolerances are inappropriate

Unlike $\mathbf{E} $, the vector potential is:

- A **numerical integral**
- Accumulated over thousands of time steps
- Sensitive to floating-point summation order

Therefore, tolerances suitable for $\mathbf{E} $
(e.g. $10^{-12}$) are **too strict** for $\mathbf{A} $.

Observed discrepancies of order:

- $10^{-3}$–$10^{-2}$ for plane waves
- $10^{-6}$–$10^{-7}$ for focused beams

are **expected and acceptable**.

---

## 9. Recommended pass/fail criteria

### Recommended defaults

```text
rtol = 1e-4
atol = 1e-8
```

This allows relative error on large values while enforcing strict absolute
accuracy when $|\mathbf{A}| $ is small.

---

## 10. Interpretation of results

- Gaussian cases often reach **machine precision**
- Hermite modes show moderate accumulated error
- Plane waves show the largest discrepancies due to non-decaying fields

This pattern is a **correctness signal**, not a problem.

---

## 11. Exit codes

- `0` — all cases passed within tolerance  
- `1` — one or more cases failed  
- `2` — runtime or configuration error  

---

## 12. Summary

- The A-comparison test validates **numerical consistency**, not absolute accuracy
- Larger tolerances than for E are **required and justified**
- Your observed results are consistent with a correct implementation

Do **not** tighten tolerances unless you also reduce the integration window or
decrease $\Delta t $.
