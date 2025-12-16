#!/usr/bin/env python3
# tests/compare_multipulse.py
#
# Compares C vs Python for a fixed 3-pulse MultiPulse:
#   - plane + elliptical
#   - gaussian + linear
#   - hermite HG(3,4) + circular
#
# C executable prints CSV rows with (t, r) samples and both E and A.
# Python rebuilds the same MultiPulse and recomputes E(t,r) and A(t,r)
# (A computed with the same trapezoid integration window and dt).
#
# Outputs per-component statistics and PASS/FAIL.

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def import_local_laserion(project_root: Path):
    sys.path.insert(0, str(project_root / "src"))
    from laserion.core import SinglePulse, MultiPulse
    from laserion.temporal_profile import GaussianTemporal
    from laserion.transverse_profile import PlaneWaveProfile, GaussianTransverse, HermiteTransverse
    from laserion.polarization import Polarization, LinearPolarization, CircularPolarization

    return SinglePulse, MultiPulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization, LinearPolarization, CircularPolarization


def stats(label: str, x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return dict(
        label=label,
        mean=float(np.mean(x)),
        rms=float(np.sqrt(np.mean(x * x))),
        max=float(np.max(np.abs(x))),
    )


def build_basis_like_c(k_vec):
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


def A_trap_from_pulse_E(pulse, t_fs, r_um, tmin_fs, tmax_fs, dt_fs):
    """Match C pulse_A_trap exactly: A(t) = -∫ E dt on uniform grid, clamped to [tmin,tmax]."""
    if not (dt_fs > 0.0 and tmax_fs > tmin_fs):
        return np.zeros(3, dtype=float)

    t_end = float(t_fs)
    if t_end < tmin_fs:
        t_end = tmin_fs
    if t_end > tmax_fs:
        t_end = tmax_fs

    n_steps = int(np.floor((t_end - tmin_fs) / dt_fs))

    E_prev = np.asarray(pulse.E(tmin_fs, r_um), dtype=float)
    A = np.zeros(3, dtype=float)

    for i in range(1, n_steps + 1):
        ti = tmin_fs + i * dt_fs
        E_cur = np.asarray(pulse.E(ti, r_um), dtype=float)
        A -= 0.5 * (E_prev + E_cur) * dt_fs
        E_prev = E_cur

    t_reached = tmin_fs + n_steps * dt_fs
    dt_last = t_end - t_reached
    if dt_last > 0.0:
        E_cur = np.asarray(pulse.E(t_end, r_um), dtype=float)
        A -= 0.5 * (E_prev + E_cur) * dt_last

    return A


def build_python_multipulse(SinglePulse, MultiPulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization, LinearPolarization, CircularPolarization):
    # Pulse 1
    temporal_1 = GaussianTemporal(tau=950.0)
    transverse_1 = PlaneWaveProfile()
    k1 = (1.0, 2.0, 3.0)
    r1s = (-25.0, 10.0, 0.0)

    # Elliptical via general Jones-based Polarization (match C: basis from linear + override)
    e1, e2 = build_basis_like_c(k1)
    e1r, e2r = rotate_in_transverse_plane(e1, e2, 20.0)
    pol1 = Polarization(e1r, e2r, p1=1.0, p2=0.55, delta=0.7)

    p1 = SinglePulse(
        E_0=120.0,
        wavelength=10.0,
        temporal=temporal_1,
        transverse=transverse_1,
        polarization=pol1,
        phase_0=0.15,
        k_vec=k1,
        use_retarded_time=True,
        r_start=r1s,
    )

    # Pulse 2
    temporal_2 = GaussianTemporal(tau=780.0)
    transverse_2 = GaussianTransverse(w0=7.0, zf=60.0)
    k2 = (0.6, 1.0, 1.2)
    r2s = (15.0, -20.0, 30.0)
    pol2 = LinearPolarization(k_vec=k2, angle=-10.0)

    p2 = SinglePulse(
        E_0=180.0,
        wavelength=9.0,
        temporal=temporal_2,
        transverse=transverse_2,
        polarization=pol2,
        phase_0=-0.25,
        k_vec=k2,
        use_retarded_time=True,
        r_start=r2s,
    )

    # Pulse 3
    temporal_3 = GaussianTemporal(tau=1100.0)
    transverse_3 = HermiteTransverse(w0=6.0, zf=-40.0, l=3, m=4)
    k3 = (1.2, 0.3, 1.0)
    r3s = (35.0, 15.0, -10.0)
    pol3 = CircularPolarization(k_vec=k3, sense="right", angle=35.0)

    p3 = SinglePulse(
        E_0=150.0,
        wavelength=10.5,
        temporal=temporal_3,
        transverse=transverse_3,
        polarization=pol3,
        phase_0=0.05,
        k_vec=k3,
        use_retarded_time=True,
        r_start=r3s,
    )

    return MultiPulse([p1, p2, p3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-exe", required=True, help="Path to csrc/bin/test_multipulse")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--tspan", type=float, default=2000.0)
    ap.add_argument("--xspan", type=float, default=50.0)
    ap.add_argument("--yspan", type=float, default=50.0)
    ap.add_argument("--zspan", type=float, default=50.0)

    ap.add_argument("--Atmin", type=float, required=True)
    ap.add_argument("--Atmax", type=float, required=True)
    ap.add_argument("--Adt", type=float, required=True)

    ap.add_argument("--E_rtol", type=float, default=1e-12)
    ap.add_argument("--E_atol", type=float, default=1e-12)

    ap.add_argument("--A_rtol", type=float, default=1e-4)
    ap.add_argument("--A_atol", type=float, default=1e-8)

    ap.add_argument("--print-frac", type=float, default=0.0)

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    SinglePulse, MultiPulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization, LinearPolarization, CircularPolarization = import_local_laserion(root)

    cmd = [
        args.c_exe,
        "--N",
        str(args.N),
        "--seed",
        str(args.seed),
        "--tspan",
        str(args.tspan),
        "--xspan",
        str(args.xspan),
        "--yspan",
        str(args.yspan),
        "--zspan",
        str(args.zspan),
        "--Atmin",
        str(args.Atmin),
        "--Atmax",
        str(args.Atmax),
        "--Adt",
        str(args.Adt),
        "--timing",
        "1",
    ]

    tC0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tC1 = time.perf_counter()

    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)

    for ln in res.stderr.splitlines():
        if ln.strip().startswith("[C timing]"):
            print(ln)
    print(f"[Python timing] C subprocess wall time: {tC1 - tC0:.6f} s")

    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    header = lines[0].split(",")
    data = np.genfromtxt(lines[1:], delimiter=",")
    if data.ndim == 1:
        data = data[None, :]

    col = {name: j for j, name in enumerate(header)}

    t = data[:, col["t_fs"]]
    r = np.column_stack([data[:, col["x_um"]], data[:, col["y_um"]], data[:, col["z_um"]]])
    E_c = np.column_stack([data[:, col["Ex"]], data[:, col["Ey"]], data[:, col["Ez"]]])
    A_c = np.column_stack([data[:, col["Ax"]], data[:, col["Ay"]], data[:, col["Az"]]])

    mp = build_python_multipulse(SinglePulse, MultiPulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization, LinearPolarization, CircularPolarization)

    t0 = time.perf_counter()
    E_py = np.array([mp.E(tt, rr) for tt, rr in zip(t, r)], dtype=float)
    t1 = time.perf_counter()
    print(f"[Python timing] E evaluations: {t1 - t0:.6f} s for N={len(t)}")

    t2 = time.perf_counter()
    A_py = np.array([A_trap_from_pulse_E(mp, tt, rr, args.Atmin, args.Atmax, args.Adt) for tt, rr in zip(t, r)], dtype=float)
    t3 = time.perf_counter()
    print(f"[Python timing] A integrations: {t3 - t2:.6f} s for N={len(t)}")

    dE = E_c - E_py
    dA = A_c - A_py

    max_dE = float(np.max(np.abs(dE)))
    max_dA = float(np.max(np.abs(dA)))

    print("\n--- E differences (C - Python) ---")
    for k in range(3):
        s = stats(f"dE[{k}]", dE[:, k])
        print(f"  {s['label']}: mean={s['mean']:.3e}  rms={s['rms']:.3e}  max={s['max']:.3e}")
    print(f"  max |dE| = {max_dE:.3e}  -> {'PASS' if np.allclose(E_c, E_py, rtol=args.E_rtol, atol=args.E_atol) else 'FAIL'}")

    print("\n--- A differences (C - Python) ---")
    for k in range(3):
        s = stats(f"dA[{k}]", dA[:, k])
        print(f"  {s['label']}: mean={s['mean']:.3e}  rms={s['rms']:.3e}  max={s['max']:.3e}")
    print(f"  max |dA| = {max_dA:.3e}  -> {'PASS' if np.allclose(A_c, A_py, rtol=args.A_rtol, atol=args.A_atol) else 'FAIL'}")

    if args.print_frac and args.print_frac > 0:
        n = len(t)
        step = max(1, int(round(1.0 / args.print_frac)))
        idx = np.arange(0, n, step, dtype=int)
        print("\n--- Sample rows (subset) ---")
        print("i,t_fs,x_um,y_um,z_um,Ex_c,Ey_c,Ez_c,Ex_py,Ey_py,Ez_py,Ax_c,Ay_c,Az_c,Ax_py,Ay_py,Az_py")
        for ii in idx:
            row = [
                int(ii),
                t[ii],
                r[ii, 0],
                r[ii, 1],
                r[ii, 2],
                *E_c[ii],
                *E_py[ii],
                *A_c[ii],
                *A_py[ii],
            ]
            print(",".join(f"{v:.17g}" if isinstance(v, float) else str(v) for v in row))

    ok = np.allclose(E_c, E_py, rtol=args.E_rtol, atol=args.E_atol) and np.allclose(A_c, A_py, rtol=args.A_rtol, atol=args.A_atol)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
