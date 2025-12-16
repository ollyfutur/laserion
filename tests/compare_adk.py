#!/usr/bin/env python3
# tests/compare_adk.py
#
# Exhaustive ADK comparison:
#   - all species
#   - all charge states
#   - E = 10^k sweep
#
# Optional verbose printing of C/Python values.

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np


def import_local_laserion(root: Path):
    sys.path.insert(0, str(root / "src"))
    from laserion.ionization import ADKModel
    return ADKModel()


def parse_csv(txt):
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if lines[0] != "species,Z,E_abs,ion_ene_eV,w":
        raise RuntimeError("Bad CSV header")

    rows = []
    for ln in lines[1:]:
        s, Z, E, Ei, w = ln.split(",")
        rows.append((s, int(Z), float(E), float(Ei), float(w)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Emin-exp", type=int, default=0)
    ap.add_argument("--Emax-exp", type=int, default=9)
    ap.add_argument("--c-exe", required=True)

    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-15)
    ap.add_argument("--logtol", type=float, default=1e-10)
    ap.add_argument("--log-floor", type=float, default=1e-300)

    ap.add_argument("--print-all", action="store_true",
                    help="Print C and Python rates for all cases.")
    ap.add_argument("--print-mismatch", action="store_true", default=True,
                    help="Print only mismatching cases (default).")

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    adk = import_local_laserion(root)

    res = subprocess.run(
        [
            args.c_exe,
            "--Emin-exp", str(args.Emin_exp),
            "--Emax-exp", str(args.Emax_exp),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if res.returncode != 0:
        print(res.stderr)
        sys.exit(2)

    rows = parse_csv(res.stdout)

    failures = 0

    if args.print_all:
        print("species  Z   E        w_C            w_PY           dlog")

    for species, Z, E, Ei_c, w_c in rows:
        Ei_py = adk.ionization_energy(species, Z)
        w_py = adk.rate(E, species, Z)

        # Ionization energy check
        if abs(Ei_py - Ei_c) > 1e-12:
            print(f"[ION_E] {species} Z={Z}: C={Ei_c} PY={Ei_py}")
            failures += 1
            continue

        # Linear comparison
        lin_ok = np.isclose(w_c, w_py, rtol=args.rtol, atol=args.atol)

        # Log comparison (robust)
        wc = max(w_c, args.log_floor)
        wp = max(w_py, args.log_floor)
        dlog = abs(np.log(wc) - np.log(wp))
        log_ok = dlog <= args.logtol

        ok = lin_ok or log_ok

        if args.print_all or (args.print_mismatch and not ok):
            print(
                f"{species:>3s} {Z:2d} {E:7.1e} "
                f"{w_c:14.6e} {w_py:14.6e} {dlog:9.2e}"
            )

        if not ok:
            failures += 1

    if failures == 0:
        print("ADK TEST PASSED: all species, all Z, all fields")
        sys.exit(0)
    else:
        print(f"ADK TEST FAILED: {failures} mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()

