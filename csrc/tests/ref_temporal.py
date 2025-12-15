#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from pathlib import Path

# -------------------------------------------------
# Ensure we import *your* laserion code
# -------------------------------------------------
# Adjust if needed, but this is the usual layout:
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
    ap.add_argument("--out", type=str, default="out_temporal_py.csv")
    args = ap.parse_args()

    tau, tmin, tmax, dt = args.tau, args.tmin, args.tmax, args.dt
    if not (tau > 0 and tmax > tmin and dt > 0):
        raise ValueError("Invalid parameters.")

    temporal = GaussianTemporal(tau)

    t = np.arange(tmin, tmax + 0.5 * dt, dt, dtype=float)
    env = temporal(t)

    data = np.column_stack([t, env])
    header = "t_fs,env"
    np.savetxt(
        args.out,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.17g",
    )


if __name__ == "__main__":
    main()

