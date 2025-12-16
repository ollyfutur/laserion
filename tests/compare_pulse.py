#!/usr/bin/env python3
# tests/compare_pulse.py
#
# Runs 3x3 SinglePulse comparisons:
#   Transverse: plane, gaussian, hermite (higher-order)
#   Polarization: linear, circular, elliptical (arbitrary delta)
#
# Compares Ex,Ey,Ez point-by-point on the same sampled (t,x,y,z) from C.
# Optional: print a fraction of sampled values to screen.

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def import_local_laserion(project_root: Path):
    sys.path.insert(0, str(project_root / "src"))
    from laserion.core import SinglePulse
    from laserion.temporal_profile import GaussianTemporal
    from laserion.transverse_profile import PlaneWaveProfile, GaussianTransverse, HermiteTransverse
    from laserion.polarization import Polarization

    return SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization


def build_basis_like_c(k_vec):
    """
    Reproduce the exact basis construction used by C LinearPolarization_init:
      - k_hat = k / |k|
      - if k not parallel z: e1 = z - (z·k)k  (normalized)
        else e1 = x
      - e2 = cross(k_hat, e1) (normalized)
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


def polarization_elliptical(PolarizationCls, k_vec, p1, p2, delta_rad, angle_deg):
    e1, e2 = build_basis_like_c(k_vec)
    e1r, e2r = rotate_in_transverse_plane(e1, e2, angle_deg)
    return PolarizationCls(e1r, e2r, p1=float(p1), p2=float(p2), delta=float(delta_rad))


def parse_c_csv(txt):
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines or lines[0].lower() != "i,t_fs,x_um,y_um,z_um,ex,ey,ez":
        raise RuntimeError("Bad C CSV header for pulse test.")
    data = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != 8:
            raise RuntimeError(f"Bad CSV line: {ln!r}")
        i = int(parts[0])
        t = float(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        ex = float(parts[5])
        ey = float(parts[6])
        ez = float(parts[7])
        data.append((i, t, x, y, z, ex, ey, ez))
    arr = np.array(data, dtype=float)
    return arr


def print_fraction(samples, frac, title):
    if frac <= 0:
        return
    n = samples.shape[0]
    k = max(1, int(np.floor(frac * n)))
    idx = np.linspace(0, n - 1, k, dtype=int)
    print(title)
    print("  i      t_fs       x_um       y_um       z_um        Ex(C)        Ey(C)        Ez(C)        Ex(PY)       Ey(PY)       Ez(PY)")
    for j in idx:
        row = samples[j]
        # row layout: [t,x,y,z, ExC,EyC,EzC, ExP,EyP,EzP]
        print(f"{j:4d} {row[0]:9.3e} {row[1]:9.3e} {row[2]:9.3e} {row[3]:9.3e} " f"{row[4]:12.4e} {row[5]:12.4e} {row[6]:12.4e} " f"{row[7]:12.4e} {row[8]:12.4e} {row[9]:12.4e}")
    print("")


def stats(name, d):
    d = np.asarray(d, dtype=float)
    return f"{name}: mean={d.mean():.3e}  rms={np.sqrt(np.mean(d*d)):.3e}  max={np.max(np.abs(d)):.3e}"


def run_case(case, args, SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization):
    c_exe = Path(args.c_exe)
    if not c_exe.exists():
        raise FileNotFoundError(f"C executable not found: {c_exe}")

    # Run C to produce (t,x,y,z,Ex,Ey,Ez)
    cmd = [
        str(c_exe),
        "--N",
        str(args.N),
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

    # Optional sampling overrides
    if case.get("tspan") is not None:
        cmd += ["--tspan", str(case["tspan"])]
    if case.get("xspan") is not None:
        cmd += ["--xspan", str(case["xspan"])]
    if case.get("yspan") is not None:
        cmd += ["--yspan", str(case["yspan"])]
    if case.get("zspan") is not None:
        cmd += ["--zspan", str(case["zspan"])]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"C failed.\nCommand: {' '.join(cmd)}\n\nstderr:\n{res.stderr}")

    arr = parse_c_csv(res.stdout)
    # columns: i,t,x,y,z,ExC,EyC,EzC
    t = arr[:, 1]
    r = arr[:, 2:5]
    E_c = arr[:, 5:8]

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
        pol = polarization_linear(Polarization, case["k_vec"], case["angle_deg"])
    elif case["pol_kind"] == "circular":
        pol = polarization_circular(Polarization, case["k_vec"], case["sense"], case["angle_deg"])
    elif case["pol_kind"] == "elliptical":
        pol = polarization_elliptical(Polarization, case["k_vec"], case["p1"], case["p2"], case["delta"], case["angle_deg"])
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

    E_py = np.array([pulse.E(tt, rr) for tt, rr in zip(t, r)], dtype=float)

    dE = E_c - E_py
    # Prepare a combined array for printing fractions
    combined = np.column_stack([t, r, E_c, E_py])

    if args.print_frac > 0:
        print_fraction(combined, args.print_frac, title=f"Case: {case['name']}")

    # Stats per component
    out = {
        "name": case["name"],
        "max_abs": float(np.max(np.abs(dE))),
        "dx": stats("dEx", dE[:, 0]),
        "dy": stats("dEy", dE[:, 1]),
        "dz": stats("dEz", dE[:, 2]),
        "ok": np.allclose(E_c, E_py, rtol=args.rtol, atol=args.atol),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-exe", required=True, help="Path to csrc/bin/test_pulse")
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)

    ap.add_argument("--print-frac", type=float, default=0.0, help="If >0, print this fraction of sampled rows (evenly spaced). Example: 0.01 prints ~1%%.")

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    SinglePulse, GaussianTemporal, PlaneWaveProfile, GaussianTransverse, HermiteTransverse, Polarization = import_local_laserion(root)

    # ----------------------------
    # 3 (transverse) x 3 (pol) cases
    # ----------------------------
    base = dict(
        E0=150.0,
        wavelength=10.0,
        phase0=0.3,
        k_vec=[3.0, 2.0, 1.0],
        r_start=[0.0, 0.0, 0.0],
        retarded=True,
        # optional overrides: tspan/xspan/yspan/zspan
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

    # One temporal profile: GaussianTemporal only; vary tau slightly per case-set
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
            print(f"  max |dE| = {r['max_abs']:.3e}  -> {'PASS' if r['ok'] else 'FAIL'}")
            print("")
            if not r["ok"]:
                fail = True

    print("Summary:")
    n_fail = sum(0 if r["ok"] else 1 for r in results)
    print(f"  cases: {len(results)}  failed: {n_fail}")

    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
