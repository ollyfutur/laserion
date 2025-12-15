#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def import_local_laserion(project_root: Path):
    # Force using local package sources at <root>/src
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))
    from laserion.temporal_profile import GaussianTemporal  # noqa
    return GaussianTemporal


def parse_c_csv_stdout(txt: str):
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines or lines[0].lower() != "t_fs,env":
        raise RuntimeError("C output did not start with expected CSV header 't_fs,env'")
    t = []
    env = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != 2:
            raise RuntimeError(f"Bad CSV line from C: {ln!r}")
        t.append(float(parts[0]))
        env.append(float(parts[1]))
    return np.array(t, dtype=float), np.array(env, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--tmin", type=float, required=True)
    ap.add_argument("--tmax", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--c-exe", type=str, required=True)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--plot", action="store_true", help="Show an overlay plot (Python vs C).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    GaussianTemporal = import_local_laserion(project_root)

    c_exe = Path(args.c_exe)
    if not c_exe.exists():
        raise FileNotFoundError(f"C executable not found: {c_exe}")

    # Run C and capture CSV from stdout
    cmd = [
        str(c_exe),
        "--tau", str(args.tau),
        "--tmin", str(args.tmin),
        "--tmax", str(args.tmax),
        "--dt", str(args.dt),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "C executable failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stderr:\n{res.stderr}"
        )

    t_c, env_c = parse_c_csv_stdout(res.stdout)

    # Python reference on the same times
    temporal = GaussianTemporal(tau=args.tau)
    env_py = temporal(t_c)

    # Compare
    ok = np.allclose(env_c, env_py, rtol=args.rtol, atol=args.atol)

    diff = np.abs(env_c - env_py)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / np.maximum(np.abs(env_py), 1e-300)).max()) if diff.size else 0.0

    print(f"Temporal Gaussian comparison")
    print(f"  tau={args.tau} fs, t=[{args.tmin},{args.tmax}] fs, dt={args.dt} fs, N={t_c.size}")
    print(f"  max_abs={max_abs:.3e}")
    print(f"  max_rel={max_rel:.3e}")
    print(f"  allclose(rtol={args.rtol}, atol={args.atol}) -> {ok}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t_c, env_py, label="Python")
        plt.plot(t_c, env_c, linestyle="--", markersize=3, label="C")
        plt.xlabel("t [fs]")
        plt.ylabel("envelope")
        plt.grid(True)
        plt.legend()
        plt.title("GaussianTemporal: C vs Python")
        plt.show()

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

