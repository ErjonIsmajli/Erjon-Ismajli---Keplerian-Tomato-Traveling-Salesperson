"""Evaluate every solution file in solutions/ and print the Markdown results table of the README."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kttsp.evaluator import evaluate  # noqa: E402
from kttsp.instance import load_instance  # noqa: E402
from kttsp.solution import Solution  # noqa: E402

REFERENCE = {"easy": 116.911, "medium": 234.929, "hard": 393.229}   # ScienceFlow entry, arXiv:2608.14354


def main():
    print("| instance | tomatoes | mission time [days] | exceptions used | max delta-v [m/s] | solver time | "
          "reference entry [days] |")
    print("|---|---|---|---|---|---|---|")
    for name in ["easy", "medium", "hard"]:
        path = ROOT / "solutions" / f"{name}.json"
        if not path.exists():
            continue
        inst = load_instance(ROOT / "data" / f"{name}.kttsp")
        sol = Solution.load(path)
        rep = evaluate(inst, sol.to_chromosome())
        secs = sol.meta.get("total_seconds") or (sol.meta.get("construct_seconds", 0)
                                                 + sol.meta.get("improve_seconds", 0))
        status = "" if rep.feasible else " (INFEASIBLE)"
        print(f"| {name} | {inst.n} | {rep.objective:.3f}{status} | {rep.n_exceptions} / {inst.n_exceptions} | "
              f"{max(leg.dv for leg in rep.legs):.2f} | {secs / 60:.0f} min | {REFERENCE[name]:.1f} |")
        for v in rep.violations:
            print("   ", v)


if __name__ == "__main__":
    main()
