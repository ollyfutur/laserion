import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def import_local_laserion(project_root: Path):
    # Force using local package sources at <root>/src
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))

    from laserion.transverse_profile import PlaneWaveProfile, GaussianTransverse, HermiteTransverse  # noqa

    return PlaneWaveProfile, GaussianTransverse, HermiteTransverse


def parse_c_csv_stdout(txt: str):
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines or lines[0].lower() != "x_um,y_um,z_um,env,phase":
        raise RuntimeError("C output did not start with expected CSV header 'x_um,y_um,z_um,env,phase'")

    x = np.empty(len(lines) - 1, dtype=float)
    y = np.empty(len(lines) - 1, dtype=float)
    z = np.empty(len(lines) - 1, dtype=float)
    env = np.empty(len(lines) - 1, dtype=float)
    ph = np.empty(len(lines) - 1, dtype=float)

    for i, ln in enumerate(lines[1:]):
        parts = ln.split(",")
        if len(parts) != 5:
            raise RuntimeError(f"Bad CSV line from C: {ln!r}")
        x[i] = float(parts[0])
        y[i] = float(parts[1])
        z[i] = float(parts[2])
        env[i] = float(parts[3])
        ph[i] = float(parts[4])

    return x, y, z, env, ph


def wrap_to_pi(dphi):
    # map to (-pi, pi]
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def stats(name, arr):
    arr = np.asarray(arr, dtype=float)
    return (
        f"{name}: mean={arr.mean():.3e}  rms={np.sqrt((arr*arr).mean()):.3e}  max={arr.max():.3e}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["plane", "gaussian", "hermite"], required=True)
    ap.add_argument("--wavelength", type=float, required=True)
    ap.add_argument("--w0", type=float, default=None)
    ap.add_argument("--zf", type=float, default=0.0)
    ap.add_argument("--l", type=int, default=0)
    ap.add_argument("--m", type=int, default=0)

    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--xspan", type=float, default=None)
    ap.add_argument("--yspan", type=float, default=None)
    ap.add_argument("--zspan", type=float, default=None)

    ap.add_argument("--c-exe", type=str, required=True)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)

    ap.add_argument("--print-all", action="store_true", help="Print per-point diffs for all points.")
    ap.add_argument("--print-max", type=int, default=50, help="If not --print-all, print at most this many points.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    PlaneWaveProfile, GaussianTransverse, HermiteTransverse = import_local_laserion(project_root)

    c_exe = Path(args.c_exe)
    if not c_exe.exists():
        raise FileNotFoundError(f"C executable not found: {c_exe}")

    # Validate required parameters
    if args.profile != "plane" and (args.w0 is None or args.w0 <= 0.0):
        raise SystemExit("For gaussian/hermite, you must provide --w0 > 0.")

    # Call C to generate random points + C results (CSV on stdout)
    cmd = [
        str(c_exe),
        "--profile", args.profile,
        "--wavelength", str(args.wavelength),
        "--N", str(args.N),
        "--seed", str(args.seed),
    ]
    if args.profile != "plane":
        cmd += ["--w0", str(args.w0), "--zf", str(args.zf)]
        if args.profile == "hermite":
            cmd += ["--l", str(args.l), "--m", str(args.m)]

    if args.xspan is not None:
        cmd += ["--xspan", str(args.xspan)]
    if args.yspan is not None:
        cmd += ["--yspan", str(args.yspan)]
    if args.zspan is not None:
        cmd += ["--zspan", str(args.zspan)]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "C executable failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stderr:\n{res.stderr}"
        )

    x, y, z, env_c, ph_c = parse_c_csv_stdout(res.stdout)

    # Build Python profile
    if args.profile == "plane":
        prof = PlaneWaveProfile()
    elif args.profile == "gaussian":
        prof = GaussianTransverse(w0=float(args.w0), zf=float(args.zf))
    else:
        prof = HermiteTransverse(w0=float(args.w0), zf=float(args.zf), l=int(args.l), m=int(args.m))

    # Python evaluation (beam-frame points)
    r = np.vstack([x, y, z]).T  # (N,3)

    # Python code expects: prof(r_i, wavelength) and prof.phase(r_i, wavelength)
    env_py = np.array([prof(ri, args.wavelength) for ri in r], dtype=float)
    ph_py = np.array([prof.phase(ri, args.wavelength) for ri in r], dtype=float)

    # Differences
    denv = env_c - env_py
    dphi = wrap_to_pi(ph_c - ph_py)

    abs_env = np.abs(denv)
    abs_phi = np.abs(dphi)

    # Print per-point differences
    nprint = args.N if args.print_all else min(args.N, args.print_max)

    print("Transverse profile comparison (beam frame)")
    print(f"  profile={args.profile}")
    print(f"  wavelength={args.wavelength} um, N={args.N}, seed={args.seed}")
    if args.profile != "plane":
        print(f"  w0={args.w0} um, zf={args.zf} um" + (f", l={args.l}, m={args.m}" if args.profile == "hermite" else ""))

    print("")
    print("Per-point differences:")
    print("  i   x        y        z        env_C      env_PY     d_env      ph_C       ph_PY      dphi_wrapped")
    for i in range(nprint):
        print(
            f"{i:4d} "
            f"{x[i]: .4e} {y[i]: .4e} {z[i]: .4e} "
            f"{env_c[i]: .6e} {env_py[i]: .6e} {denv[i]: .3e} "
            f"{ph_c[i]: .6e} {ph_py[i]: .6e} {dphi[i]: .3e}"
        )
    if not args.print_all and args.N > nprint:
        print(f"... (printed {nprint}/{args.N}; use --print-all to print everything)")

    # Summary statistics
    print("\nSummary statistics:")
    print("  Envelope |C - PY| : " + stats("abs", abs_env))
    print("  Phase    |wrap(C - PY)| : " + stats("abs", abs_phi))

    # Pass/fail using allclose on env and wrapped phase
    ok_env = np.allclose(env_c, env_py, rtol=args.rtol, atol=args.atol)
    # for phase: compare wrapped difference to tolerances
    ok_phi = np.all(np.abs(dphi) <= (args.atol + args.rtol * np.maximum(np.abs(ph_py), 1.0)))

    print(f"\nChecks:")
    print(f"  envelope allclose(rtol={args.rtol}, atol={args.atol}) -> {ok_env}")
    print(f"  phase wrapped tolerance (rtol={args.rtol}, atol={args.atol}) -> {ok_phi}")

    raise SystemExit(0 if (ok_env and ok_phi) else 1)


if __name__ == "__main__":
    main()

