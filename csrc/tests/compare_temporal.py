#!/usr/bin/env python3
import argparse
import subprocess
import numpy as np
from pathlib import Path

def load_csv(path: Path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return arr["t_fs"], arr["env"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=60.0)
    ap.add_argument("--tmin", type=float, default=-200.0)
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--c-exe", type=str, default="csrc/test_temporal")
    ap.add_argument("--outdir", type=str, default="tests/_out")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-14)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_c = outdir / "out_temporal_c.csv"
    out_p = outdir / "out_temporal_py.csv"

    # 1) Run C executable
    c_exe = Path(args.c_exe)
    if not c_exe.exists():
        raise FileNotFoundError(f"C executable not found: {c_exe}")

    subprocess.check_call([
        str(c_exe),
        str(args.tau),
        str(args.tmin),
        str(args.tmax),
        str(args.dt),
        str(out_c),
    ])

    # 2) Run Python ref generator
    refgen = Path("tests/ref_temporal.py")
    if not refgen.exists():
        raise FileNotFoundError("Missing tests/ref_temporal.py")

    subprocess.check_call([
        "python3.9",
        str(refgen),
        "--tau", str(args.tau),
        "--tmin", str(args.tmin),
        "--tmax", str(args.tmax),
        "--dt", str(args.dt),
        "--out", str(out_p),
    ])

    # 3) Compare
    t_c, env_c = load_csv(out_c)
    t_p, env_p = load_csv(out_p)

    if len(t_c) != len(t_p):
        raise AssertionError(f"Grid length mismatch: C={len(t_c)} Python={len(t_p)}")

    if not np.allclose(t_c, t_p, rtol=0, atol=0):
        raise AssertionError("Time grids do not match exactly.")

    diff = np.abs(env_c - env_p)
    max_abs = float(diff.max())
    max_rel = float((diff / np.maximum(np.abs(env_p), 1e-300)).max())

    ok = np.allclose(env_c, env_p, rtol=args.rtol, atol=args.atol)
    print(f"max_abs_diff = {max_abs:.3e}")
    print(f"max_rel_diff = {max_rel:.3e}")
    print(f"tolerance: rtol={args.rtol:.3e}, atol={args.atol:.3e}")

    if not ok:
        # print a small diagnostic
        idx = int(diff.argmax())
        print("Worst point:")
        print(f"  t = {t_c[idx]:.17g}")
        print(f"  C = {env_c[idx]:.17g}")
        print(f"  P = {env_p[idx]:.17g}")
        print(f"  |d| = {diff[idx]:.3e}")
        raise AssertionError("Temporal envelope mismatch beyond tolerance.")

    print("OK: C and Python temporal envelopes match.")

if __name__ == "__main__":
    main()

