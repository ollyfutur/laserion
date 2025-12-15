#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# -------------------------------------------------
# Ensure we import *your* laserion code
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from laserion.temporal_profile import GaussianTemporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--tmin", type=float, required=True)
    ap.add_argument("--tmax", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--c-exe", type=str, required=True,
                    help="Path to C temporal test executable")
    args = ap.parse_args()

    tau = args.tau
    tmin = args.tmin
    tmax = args.tmax
    dt = args.dt
    c_exe = Path(args.c_exe)

    if not c_exe.exists():
        raise FileNotFoundError(f"C executable not found: {c_exe}")

    # -------------------------------------------------
    # 1) Run C code → temporary CSV
    # -------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        c_csv = Path(tmpdir) / "temporal_c.csv"

        subprocess.check_call([
            str(c_exe),
            str(tau),
            str(tmin),
            str(tmax),
            str(dt),
            str(c_csv),
        ])

        c_data = np.loadtxt(c_csv, delimiter=",", skiprows=1)

    t_c = c_data[:, 0]
    env_c = c_data[:, 1]

    # -------------------------------------------------
    # 2) Python reference using your package
    # -------------------------------------------------
    t_py = np.arange(tmin, tmax + 0.5 * dt, dt, dtype=float)
    temporal = GaussianTemporal(tau)
    env_py = temporal(t_py)

    # -------------------------------------------------
    # 3) Plot
    # -------------------------------------------------
    plt.figure()
    plt.plot(t_py, env_py, "-", label="Python GaussianTemporal")
    plt.plot(t_c, env_c, "o", ms=3, label="C GaussianTemporal")

    plt.xlabel("t [fs]")
    plt.ylabel("Envelope")
    plt.title(f"Temporal envelope comparison (tau = {tau} fs)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()

