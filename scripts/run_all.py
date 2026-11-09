"""Solve the three official instances, write the solution/submission files and the figures into
an output folder (runs/ by default, so the final results in solutions/ are never overwritten),
and print a Markdown table with the results.

    .venv/bin/python scripts/run_all.py --budget-easy 900 --budget-medium 2400 --budget-hard 4200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kttsp.construct import ConstructSettings, construct  # noqa: E402
from kttsp.evaluator import evaluate  # noqa: E402
from kttsp.improve import improve  # noqa: E402
from kttsp.instance import load_instance  # noqa: E402
from kttsp.kepler import Ephemeris  # noqa: E402
from kttsp.leg_search import LegSearcher, SearchSettings  # noqa: E402
from kttsp.plotting import save_all_figures  # noqa: E402

# construction settings of the published runs (the commands in section 4 of the README); the medium and
# hard tours were built before the lookahead existed, hence lookahead=0
CONSTRUCT = {
    "easy": dict(window=6, n_candidates=4, max_starts=3, start_in_high=False),
    "medium": dict(window=6, n_candidates=4, max_starts=3, lookahead=0),
    "hard": dict(window=30, n_candidates=8, max_starts=2, lookahead=0),
}
# medium is solved with the 0.01-day minimum time of flight of the GitHub copy of the file, so the
# tour is valid with both published versions of medium.kttsp
MIN_TOF_FLOOR = {"medium": 0.01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--budget-easy", type=float, default=900.0)
    ap.add_argument("--budget-medium", type=float, default=2400.0)
    ap.add_argument("--budget-hard", type=float, default=4200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default="runs", help="folder for solutions and figures (relative to the project)")
    args = ap.parse_args()
    budgets = {"easy": args.budget_easy, "medium": args.budget_medium, "hard": args.budget_hard}
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.instances:
        inst = load_instance(ROOT / "data" / f"{name}.kttsp")
        eph = Ephemeris(inst)
        searcher = LegSearcher(eph, inst, SearchSettings(min_tof_floor=MIN_TOF_FLOOR.get(name, 0.0)))
        cs = ConstructSettings(**CONSTRUCT.get(name, {}), workers=args.workers)
        t0 = time.time()
        print(f"=== {name}: {inst.n} tomatoes ===")
        sol = construct(inst, eph, searcher, cs)
        if sol is None:
            print("no feasible tour")
            continue
        out = out_dir / f"{name}.json"
        sol.meta["construct_seconds"] = round(time.time() - t0, 1)
        sol.save(out)
        sol = improve(inst, sol, eph, searcher, time_budget=budgets[name], seed=args.seed, checkpoint_path=out,
                      chains=args.workers)
        report = evaluate(inst, sol.to_chromosome(), eph, min_tof=max(inst.min_tof, MIN_TOF_FLOOR.get(name, 0.0)))
        sol.objective = report.objective
        sol.meta["total_seconds"] = round(time.time() - t0, 1)
        sol.save(out)
        sol.write_submission(out.with_name(f"submission_{name}.json"), name=f"{name} tour",
                             description=f"mission time {report.objective:.4f} days")
        print(report.summary())
        if not args.no_figures:
            save_all_figures(inst, sol, out_dir / "figures")
        rows.append((name, inst.n, report.objective, f"{report.n_exceptions} / {inst.n_exceptions}",
                     max(leg.dv for leg in report.legs), sol.meta["total_seconds"], report.feasible))
    print("\n| instance | tomatoes | mission time [days] | exceptions used | max delta-v [m/s] | runtime [s] "
          "| feasible |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]} | {r[4]:.2f} | {r[5]:.0f} | {r[6]} |")


if __name__ == "__main__":
    main()
