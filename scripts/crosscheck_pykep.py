"""Cross-check of the pure-NumPy implementation against pykep 2.x, the library behind ESA's evaluator.

Runs inside the Docker image built from docker/Dockerfile (pykep has no macOS wheels):

    docker build -t kttsp-check docker/
    docker run --rm -v "$PWD":/work kttsp-check                      # compare and check solutions
    docker run --rm -v "$PWD":/work kttsp-check python scripts/crosscheck_pykep.py --write-fixture

What it does: (1) compares the tomato ephemerides, (2) compares the transfer delta-v of random legs
with the official `compute_transfer` logic reimplemented with pykep and counts the legs the two judge
differently (normal, exception or not allowed), (3) evaluates every solution file in solutions/ with
the official transfer model and (4) optionally writes the reference fixture used by
tests/test_pykep_reference.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pykep as pk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kttsp.constants import MU_MOON  # noqa: E402
from kttsp.evaluator import evaluate  # noqa: E402
from kttsp.instance import load_instance  # noqa: E402
from kttsp.kepler import Ephemeris  # noqa: E402
from kttsp.solution import Solution  # noqa: E402
from kttsp.transfer import leg_dv, transfer_angle  # noqa: E402


# minimum time of flight [days] written in the copies of the files in the esa/SpOC4 GitHub repository
GITHUB_MIN_TOF = {"easy": 0.001, "medium": 0.01, "hard": 1.0 / 1440.0}


def official_planets(inst):
    """The tomato objects exactly as the official utilities build them."""
    return [pk.planet.keplerian(pk.epoch(inst.t0), [float(v) for v in row], MU_MOON, 0.0, 0.0, 0.0)
            for row in inst.elements]


def official_leg_dv(planets, i, j, t_dep, tof, max_revs=20):
    """Verbatim logic of TomatoProblem.compute_transfer from the official utils_users.py."""
    tof_sec = tof * pk.DAY2SEC
    r_i, v_i = planets[i].eph(pk.epoch(t_dep))
    r_j, v_j = planets[j].eph(pk.epoch(t_dep + tof))
    best_dv = float("inf")
    for cw in [False, True]:
        try:
            lp = pk.lambert_problem(r_i, r_j, tof_sec, MU_MOON, cw, max_revs)
        except Exception:
            continue
        for v1, v2 in zip(lp.get_v1(), lp.get_v2()):
            total = np.linalg.norm(np.array(v1) - np.array(v_i)) + np.linalg.norm(np.array(v2) - np.array(v_j))
            if total < best_dv:
                best_dv = total
    return best_dv


def compare_states(inst, planets, eph, rng, n=400):
    worst_r = worst_v = 0.0
    for _ in range(n):
        i = int(rng.integers(inst.n))
        t = float(rng.uniform(0.0, inst.t_max))
        r_ref, v_ref = planets[i].eph(pk.epoch(t))
        r, v = eph.state(i, t)
        worst_r = max(worst_r, float(np.linalg.norm(r - np.array(r_ref))))
        worst_v = max(worst_v, float(np.linalg.norm(v - np.array(v_ref))))
    return worst_r, worst_v


def sample_legs(inst, eph, rng, n):
    """Random legs; half of them with times of flight close to whole revolutions (the interesting region)."""
    legs = []
    while len(legs) < n:
        i, j = (int(v) for v in rng.choice(inst.n, 2, replace=False))
        t_dep = float(rng.uniform(0.0, 0.5 * inst.t_max))
        if rng.random() < 0.5:
            tof = float(10 ** rng.uniform(np.log10(inst.min_tof), np.log10(40.0)))
        else:
            tof = float((rng.integers(1, 21) + rng.uniform(-0.1, 0.1)) * eph.period[i])
        if tof < inst.min_tof:
            continue
        ang = float(transfer_angle(eph, i, j, t_dep, tof))
        if min(ang, np.pi - ang) < 1e-4:   # undefined transfer plane: not a meaningful comparison
            continue
        legs.append((i, j, t_dep, tof))
    return legs


def leg_class(inst, dv):
    """0: within the normal limit, 1: within the exception limit, 2: above it or no transfer at all."""
    if dv <= inst.dv_max:
        return 0
    return 1 if dv <= inst.dv_exception else 2


def compare_legs(inst, planets, eph, legs):
    rows, diffs, disagreements = [], [], 0
    for i, j, t_dep, tof in legs:
        ref = official_leg_dv(planets, i, j, t_dep, tof)
        ours = leg_dv(eph, i, j, t_dep, tof)
        rows.append([i, j, t_dep, tof, None if ref == float("inf") else ref])
        if ref == float("inf") or ours == float("inf"):
            diffs.append(0.0 if ref == ours else float("inf"))
        else:
            diffs.append(abs(ours - ref))
        if leg_class(inst, ours) != leg_class(inst, ref):   # the two models judge this leg differently
            disagreements += 1
    return rows, np.array(diffs), disagreements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", type=int, default=2000, help="random legs per instance to compare")
    ap.add_argument("--fixture-legs", type=int, default=300, help="legs per instance stored in the fixture")
    ap.add_argument("--write-fixture", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"pykep {pk.__version__}, numpy {np.__version__}")
    fixture = {}
    for name in ["easy", "medium", "hard"]:
        inst = load_instance(ROOT / "data" / f"{name}.kttsp")
        planets = official_planets(inst)
        eph = Ephemeris(inst)
        wr, wv = compare_states(inst, planets, eph, rng)
        t0 = time.time()
        legs = sample_legs(inst, eph, rng, args.legs)
        rows, diffs, disagreements = compare_legs(inst, planets, eph, legs)
        finite = diffs[np.isfinite(diffs)]
        print(f"[{name}] ephemeris: max |dr| = {wr:.3e} m, max |dv| = {wv:.3e} m/s | "
              f"{len(legs)} legs: max |dv_ours - dv_pykep| = {finite.max():.3e} m/s, "
              f"disagreeing on feasibility: {disagreements} ({time.time() - t0:.0f} s)")
        fixture[name] = {"legs": rows[: args.fixture_legs]}
    if args.write_fixture:
        out = ROOT / "tests" / "fixtures" / "pykep_reference.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fixture))
        print(f"fixture written to {out}")
    for path in sorted((ROOT / "solutions").glob("*.json")):
        if path.name.startswith("submission"):
            continue
        sol = Solution.load(path)
        inst = load_instance(ROOT / "data" / f"{sol.instance}.kttsp")
        planets = official_planets(inst)
        report = evaluate(inst, sol.to_chromosome(), dv_fn=lambda i, j, t, dt: official_leg_dv(planets, i, j, t, dt))
        max_dv = max((leg.dv for leg in report.legs), default=float("nan"))
        print(f"[{path.name}] official model: feasible={report.feasible} objective={report.objective:.6f} d "
              f"exceptions={report.n_exceptions} max dv={max_dv:.3f} m/s")
        for v in report.violations:
            print("   ", v)
        if sol.instance in GITHUB_MIN_TOF:
            tof_ok = min(sol.tof) >= GITHUB_MIN_TOF[sol.instance]
            print(f"    shortest time of flight {min(sol.tof):.5f} d; valid with the GitHub copy of the file "
                  f"(minimum {GITHUB_MIN_TOF[sol.instance]:.5f} d): {tof_ok and report.feasible}")


if __name__ == "__main__":
    main()
