#!/usr/bin/env python3
# tests/compare_pulse_A.py
#
# Compare vector potential A(t,r) between:
#   - C test executable (prints CSV rows with Ax,Ay,Az)
#   - Python laserion (compute A via the SAME trapezoid rule on a fixed grid)
#
# CSV format from C (no header):
#   i, pos_id, t, x, y, z, Ax, Ay, Az
#
# Usage example:
#   export PYTHONPATH=$PWD/src
#   python3 tests/compare_pulse_A.py --c-exe csrc/bin/test_pulse_A --Npos 40 --Nt 100 --seed 1 --print-frac 0.01

import argparse
import subprocess
import sys
from pathlib import Path
import time
import numpy as np


def import_local_laserion(project_root: Path):
    sys.path.insert(0, str(project_root / "src"))
    from laserion.core import SinglePulse
    from laserion.temporal_profile import GaussianTemporal
    from laserion.transverse_profile import PlaneWaveProfile, GaussianTransverse, HermiteTransverse
    from laserion.polarization import Polarization

    return SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization


def parse_c_csv(text: str) -> np.ndarray:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("No CSV rows received from C.")
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        rows.append([float(p) for p in parts])
    return np.asarray(rows, dtype=float)


def stats(name: str, x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    return f"{name}: mean={x.mean():.3e}  rms={np.sqrt(np.mean(x*x)):.3e}  max={np.max(np.abs(x)):.3e}"


def print_fraction(combined: np.ndarray, frac: float, title: str):
    """
    combined columns:
      pos_id, t, x, y, z, AxC,AyC,AzC, AxP,AyP,AzP
    """
    if frac <= 0:
        return
    n = combined.shape[0]
    k = max(1, int(round(frac * n)))
    idx = np.linspace(0, n - 1, k, dtype=int)

    print(f"--- {title} | printing {k}/{n} rows (~{100*frac:.2f}%) ---")
    header = "pos_id, t, x, y, z, AxC, AyC, AzC, AxP, AyP, AzP, dAx, dAy, dAz"
    print(header)
    for i in idx:
        row = combined[i]
        pos_id, t, x, y, z = row[0:5]
        Ac = row[5:8]
        Ap = row[8:11]
        d = Ac - Ap
        print(f"{int(pos_id):d}, " f"{t:.12g}, {x:.12g}, {y:.12g}, {z:.12g}, " f"{Ac[0]:.12g}, {Ac[1]:.12g}, {Ac[2]:.12g}, " f"{Ap[0]:.12g}, {Ap[1]:.12g}, {Ap[2]:.12g}, " f"{d[0]:.3e}, {d[1]:.3e}, {d[2]:.3e}")
    print("--- end ---")


# ---------- basis helpers (must match C) ----------


def build_basis_like_c(k_vec):
    """
    Match C's transverse basis construction in SinglePulse_init:
      - if k not (anti)parallel to z: e1 = z - (z·k)k, normalized; else e1=x
      - e2 = cross(k_hat, e1), normalized
    """
    k = np.asarray(k_vec, dtype=float)
    k_hat = k / np.linalg.norm(k)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(z_hat, k_hat)) < 0.999999:
        e1 = z_hat - np.dot(z_hat, k_hat) * k_hat
        e1 = e1 / np.linalg.norm(e1)
    else:
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
    e2 = np.cross(k_hat, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def rotate_in_transverse_plane(e1, e2, angle_deg):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    e1r = c * e1 + s * e2
    e2r = -s * e1 + c * e2
    return e1r, e2r


def polarization_linear(PolarizationCls, k_vec, angle_deg):
    e1, e2 = build_basis_like_c(k_vec)
    e1r, e2r = rotate_in_transverse_plane(e1, e2, angle_deg)
    return PolarizationCls(e1r, e2r, p1=1.0, p2=0.0, delta=0.0)


def polarization_circular(PolarizationCls, k_vec, sense, angle_deg):
    e1, e2 = build_basis_like_c(k_vec)
    e1r, e2r = rotate_in_transverse_plane(e1, e2, angle_deg)
    delta = +np.pi / 2.0
    if str(sense).lower() in ("left", "ccw", "counterclockwise", "counter-clockwise"):
        delta = -np.pi / 2.0
    return PolarizationCls(e1r, e2r, p1=1.0, p2=1.0, delta=float(delta))


def polarization_elliptical(PolarizationCls, k_vec, p1, p2, delta, angle_deg):
    e1, e2 = build_basis_like_c(k_vec)
    e1r, e2r = rotate_in_transverse_plane(e1, e2, angle_deg)
    return PolarizationCls(e1r, e2r, p1=float(p1), p2=float(p2), delta=float(delta))


# ---------- A via trapezoid on fixed grid (match C) ----------


def A_trap_on_grid(pulse, r, t_eval, tmin, tmax, dt):
    """
    Compute A(t,r) = -∫ E(t',r) dt' using trapezoid on a uniform grid [tmin,tmax]
    and linear interpolation to evaluate A at t_eval.
    This is the closest Python analog to the C implementation in core.c.
    """
    r = np.asarray(r, dtype=float)
    tmin = float(tmin)
    tmax = float(tmax)
    dt = float(dt)

    if dt <= 0.0 or tmax <= tmin:
        return np.zeros((len(t_eval), 3), dtype=float)

    # uniform grid (inclusive)
    n_steps = int(np.floor((tmax - tmin) / dt))
    t_grid = tmin + dt * np.arange(n_steps + 1, dtype=float)
    E_grid = np.array([pulse.E(tt, r) for tt in t_grid], dtype=float)

    # cumulative trapezoid (explicit loop; deterministic)
    A_grid = np.zeros_like(E_grid)
    for i in range(1, len(t_grid)):
        A_grid[i] = A_grid[i - 1] - 0.5 * (E_grid[i - 1] + E_grid[i]) * (t_grid[i] - t_grid[i - 1])

    # linear interpolation per component, clamp outside window
    t_eval = np.asarray(t_eval, dtype=float)
    t_clamped = np.clip(t_eval, tmin, tmax)
    A_eval = np.empty((len(t_eval), 3), dtype=float)
    for j in range(3):
        A_eval[:, j] = np.interp(t_clamped, t_grid, A_grid[:, j])
    return A_eval


def run_case(case, args, SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, PolarizationCls):
    c_exe = Path(args.c_exe)
    if not c_exe.exists():
        raise FileNotFoundError(f"C executable not found: {c_exe}")

    cmd = [
        str(c_exe),
        "--Npos",
        str(args.Npos),
        "--Nt",
        str(args.Nt),
        "--seed",
        str(args.seed),
        "--E0",
        str(case["E0"]),
        "--wavelength",
        str(case["wavelength"]),
        "--tau",
        str(case["tau"]),
        "--phase0",
        str(case["phase0"]),
        "--k",
        ",".join(map(str, case["k_vec"])),
        "--rstart",
        ",".join(map(str, case["r_start"])),
        "--retarded",
        str(int(case["retarded"])),
        "--profile",
        case["profile"],
        "--pol",
        case["pol_kind"],
        "--Atmin",
        str(args.Atmin),
        "--Atmax",
        str(args.Atmax),
        "--Adt",
        str(args.Adt),
    ]

    if case["profile"] in ("gaussian", "hermite"):
        cmd += ["--w0", str(case["w0"]), "--zf", str(case["zf"])]
    if case["profile"] == "hermite":
        cmd += ["--l", str(case["l"]), "--m", str(case["m"])]

    if case["pol_kind"] == "linear":
        cmd += ["--angle", str(case["angle_deg"])]
    elif case["pol_kind"] == "circular":
        cmd += ["--sense", str(case["sense"]), "--angle", str(case["angle_deg"])]
    elif case["pol_kind"] == "elliptical":
        cmd += ["--p1", str(case["p1"]), "--p2", str(case["p2"]), "--delta", str(case["delta"]), "--angle", str(case["angle_deg"])]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.stderr.strip():
        # Print C timing lines (and any other diagnostics) without mixing with CSV parsing
        for ln in res.stderr.splitlines():
            if ln.strip().startswith("[C timing]"):
                print(ln)

    if res.returncode != 0:
        raise RuntimeError(f"C failed.\nCommand: {' '.join(cmd)}\n\nstderr:\n{res.stderr}")

    arr = parse_c_csv(res.stdout)
    # columns: i,pos_id,t,x,y,z,AxC,AyC,AzC
    pos_id = arr[:, 1].astype(int)
    t = arr[:, 2]
    r = arr[:, 3:6]
    A_c = arr[:, 6:9]

    # Build matching Python pulse
    temporal = GaussianTemporal(tau=case["tau"])
    if case["profile"] == "plane":
        transverse = PlaneWaveProfile()
    elif case["profile"] == "gaussian":
        transverse = GaussianTransverse(w0=case["w0"], zf=case["zf"])
    elif case["profile"] == "hermite":
        transverse = HermiteTransverse(w0=case["w0"], zf=case["zf"], l=case["l"], m=case["m"])
    else:
        raise ValueError("Unknown profile")

    if case["pol_kind"] == "linear":
        pol = polarization_linear(PolarizationCls, case["k_vec"], case["angle_deg"])
    elif case["pol_kind"] == "circular":
        pol = polarization_circular(PolarizationCls, case["k_vec"], case["sense"], case["angle_deg"])
    elif case["pol_kind"] == "elliptical":
        pol = polarization_elliptical(PolarizationCls, case["k_vec"], case["p1"], case["p2"], case["delta"], case["angle_deg"])
    else:
        raise ValueError("Unknown pol_kind")

    pulse = SinglePulse(
        E_0=case["E0"],
        wavelength=case["wavelength"],
        temporal=temporal,
        transverse=transverse,
        polarization=pol,
        phase_0=case["phase0"],
        k_vec=case["k_vec"],
        use_retarded_time=case["retarded"],
        r_start=case["r_start"],
    )
    t0 = time.perf_counter()
    # Compute A efficiently: integrate once per position group
    A_py = np.zeros_like(A_c)
    for pid in np.unique(pos_id):
        mask = pos_id == pid
        rr = r[mask][0]  # constant for this group
        tt = t[mask]
        A_py[mask] = A_trap_on_grid(pulse, rr, tt, args.Atmin, args.Atmax, args.Adt)

    t1 = time.perf_counter()
    elapsed_py = t1 - t0
    print(f"[Python timing] A integration: {elapsed_py:.6f} s " f"for {len(np.unique(pos_id))} positions " f"({len(A_c)} total A evaluations)")
    dA = A_c - A_py
    combined = np.column_stack([pos_id, t, r, A_c, A_py])

    if args.print_frac > 0:
        print_fraction(combined, args.print_frac, title=f"Case: {case['name']}")

    out = {
        "name": case["name"],
        "max_abs": float(np.max(np.abs(dA))),
        "dx": stats("dAx", dA[:, 0]),
        "dy": stats("dAy", dA[:, 1]),
        "dz": stats("dAz", dA[:, 2]),
        "ok": np.allclose(A_c, A_py, rtol=args.rtol, atol=args.atol),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-exe", required=True, help="Path to csrc/bin/test_pulse_A")
    ap.add_argument("--Npos", type=int, default=40, help="Number of spatial positions.")
    ap.add_argument("--Nt", type=int, default=100, help="Number of times per position.")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-8)

    ap.add_argument("--Atmin", type=float, default=-4000.0, help="A integration window start [fs]. Must match C.")
    ap.add_argument("--Atmax", type=float, default=+4000.0, help="A integration window end [fs]. Must match C.")
    ap.add_argument("--Adt", type=float, default=0.01, help="A integration step [fs]. Must match C.")

    ap.add_argument("--print-frac", type=float, default=0.0, help="If >0, print this fraction of sampled rows.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization = import_local_laserion(root)

    base = dict(
        E0=150.0,
        wavelength=10.0,
        phase0=0.3,
        k_vec=[3.0, 2.0, 1.0],  # oblique to exercise basis + profiles
        r_start=[0.0, 0.0, 0.0],
        retarded=True,
    )

    transverse_cases = [
        dict(profile="plane", name="plane"),
        dict(profile="gaussian", name="gaussian", w0=5.0, zf=0.0),
        dict(profile="hermite", name="hermite_HG21", w0=5.0, zf=150.0, l=2, m=1),
    ]

    pol_cases = [
        dict(pol_kind="linear", name="linear", angle_deg=25.0),
        dict(pol_kind="circular", name="circular_R", sense="right", angle_deg=15.0),
        dict(pol_kind="elliptical", name="elliptical", p1=1.0, p2=0.6, delta=0.7, angle_deg=24.0),
    ]

    taus = [900.0, 1000.0, 1100.0]

    results = []
    fail = False
    idx_tau = 0

    for T in transverse_cases:
        for P in pol_cases:
            tau = taus[idx_tau % len(taus)]
            idx_tau += 1

            case = dict(base)
            case.update(T)
            case.update(P)
            case["tau"] = tau
            case["name"] = f"{T['name']} x {P['name']} (tau={tau:g} fs)"

            r = run_case(case, args, SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization)
            results.append(r)

            print(f"[{r['name']}]")
            print(f"  {r['dx']}")
            print(f"  {r['dy']}")
            print(f"  {r['dz']}")
            print(f"  max |dA| = {r['max_abs']:.3e}  -> {'PASS' if r['ok'] else 'FAIL'}")
            print("")
            if not r["ok"]:
                fail = True

    print("Summary:")
    n_fail = sum(0 if r["ok"] else 1 for r in results)
    print(f"  cases: {len(results)}  failed: {n_fail}")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
