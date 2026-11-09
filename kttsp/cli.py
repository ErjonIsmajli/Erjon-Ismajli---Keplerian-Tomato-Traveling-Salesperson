"""Command line interface: python -m kttsp <command> ..."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .construct import ConstructSettings, analyse, construct
from .evaluator import evaluate
from .instance import InstanceError, load_instance
from .kepler import Ephemeris
from .leg_search import LegSearcher, SearchSettings
from .solution import Solution, SolutionFileError


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _non_negative(text: str) -> float:
    value = float(text)
    if not value >= 0.0:
        raise argparse.ArgumentTypeError(f"must be a number >= 0, got {text}")
    return value


def _strict_min_tof(inst, min_tof):
    """Minimum time of flight to check against: the stricter of the file's and the requested one."""
    return inst.min_tof if min_tof is None else max(inst.min_tof, min_tof)


def cmd_info(args):
    inst = load_instance(args.instance)
    st = analyse(inst)
    print(f"instance {inst.name}: {inst.n} tomatoes, t0 = {inst.t0} d, min tof = {inst.min_tof} d, "
          f"t_max = {inst.t_max} d, dv_max = {inst.dv_max} m/s, "
          f"exception = {inst.dv_exception} m/s x {inst.n_exceptions}")
    for name, idx in (("low", st.low), ("high", st.high)):
        if idx.size:
            print(f"  {name} group: {idx.size} tomatoes, a = {inst.a[idx].mean() / 1e3:.0f} km, "
                  f"period = {inst.period()[idx].mean():.3f} d, v_circ = {inst.v_circ()[idx].mean():.0f} m/s, "
                  f"inclination {inst.inc[idx].min():.3f} .. {inst.inc[idx].max():.3f} rad")
    for k, c in enumerate(st.low_clusters):
        print(f"  low cluster {k}: {c.size} tomatoes, "
              f"inclination {inst.inc[c].min():.3f} .. {inst.inc[c].max():.3f} rad")
    if st.high_chain.size >= 2:
        gaps = np.diff(inst.inc[st.high_chain])
        limit = inst.max_plane_change(inst.dv_max, inst.v_circ()[st.high].mean())
        print(f"  high chain: largest inclination gap {gaps.max():.4f} rad "
              f"(limit for {inst.dv_max:.0f} m/s: {limit:.4f} rad)")


def cmd_solve(args):
    inst = load_instance(args.instance)
    eph = Ephemeris(inst)
    searcher = LegSearcher(eph, inst, SearchSettings(min_tof_floor=args.min_tof or 0.0))
    cs = ConstructSettings(window=args.window, n_candidates=args.candidates, max_starts=args.starts,
                           workers=args.workers, lookahead=args.lookahead,
                           start_in_high=not args.no_start_high, verbose=args.verbose)
    t0 = time.time()
    sol = construct(inst, eph, searcher, cs)
    if sol is None:
        print("no feasible tour found")
        sys.exit(1)
    sol.meta["construct_seconds"] = round(time.time() - t0, 1)
    if args.budget > 0:
        from .improve import improve
        sol = improve(inst, sol, eph, searcher, time_budget=args.budget, seed=args.seed, log=print,
                      chains=args.chains)
    report = evaluate(inst, sol.to_chromosome(), eph, min_tof=_strict_min_tof(inst, args.min_tof))
    sol.objective = report.objective
    sol.meta["total_seconds"] = round(time.time() - t0, 1)
    print(report.summary())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sol.save(out)
        sol.write_submission(out.with_name("submission_" + out.name), name=f"{inst.name} tour",
                             description=f"mission time {report.objective:.4f} days")
        print(f"saved {out} and {out.with_name('submission_' + out.name)}")
    print(f"total time {time.time() - t0:.0f} s")


def cmd_improve(args):
    from .improve import improve
    inst = load_instance(args.instance)
    sol = Solution.load(args.solution)
    # time already spent on this tour, read before improve() adds this run to improve_seconds
    previous = sol.meta.get("total_seconds") or (sol.meta.get("construct_seconds", 0)
                                                 + sol.meta.get("improve_seconds", 0))
    eph = Ephemeris(inst)
    searcher = LegSearcher(eph, inst, SearchSettings(min_tof_floor=args.min_tof or 0.0))
    out = Path(args.out or args.solution)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sol = improve(inst, sol, eph, searcher, time_budget=args.budget, seed=args.seed, checkpoint_path=out, log=print,
                  chains=args.chains)
    report = evaluate(inst, sol.to_chromosome(), eph, min_tof=_strict_min_tof(inst, args.min_tof))
    sol.objective = report.objective
    sol.meta["total_seconds"] = round(previous + time.time() - t0, 1)
    print(report.summary())
    sol.save(out)
    sol.write_submission(out.with_name("submission_" + out.name), name=f"{inst.name} tour",
                         description=f"mission time {report.objective:.4f} days")
    print(f"saved {out} ({time.time() - t0:.0f} s)")


def cmd_evaluate(args):
    inst = load_instance(args.instance)
    sol = Solution.load(args.solution)
    report = evaluate(inst, sol.to_chromosome(), min_tof=args.min_tof)
    print(report.summary())
    sys.exit(0 if report.feasible else 2)


def cmd_plot(args):
    from .plotting import save_all_figures
    inst = load_instance(args.instance)
    sol = Solution.load(args.solution)
    paths = save_all_figures(inst, sol, Path(args.out))
    for p in paths:
        print("wrote", p)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kttsp", description="Keplerian Tomato Traveling Salesperson solver")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("info", help="describe an instance file")
    p.add_argument("instance")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("solve", help="build and improve a tour")
    p.add_argument("instance")
    p.add_argument("--out", default=None, help="solution JSON to write (a submission file is written next to it)")
    p.add_argument("--budget", type=_non_negative, default=0.0,
                   help="seconds for the improvement phase (0 = construction only)")
    p.add_argument("--window", type=_positive_int, default=6)
    p.add_argument("--candidates", type=_positive_int, default=4)
    p.add_argument("--starts", type=_positive_int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=_positive_int, default=1, help="processes for the construction phase")
    p.add_argument("--lookahead", type=int, default=3, help="candidates judged by their best successor (0 = off)")
    p.add_argument("--chains", type=_positive_int, default=1, help="parallel annealing chains in the improvement phase")
    p.add_argument("--no-start-high", action="store_true", help="only plans that start inside a low cluster")
    p.add_argument("--min-tof", type=_non_negative, default=None,
                   help="stricter minimum time of flight [days] than the instance file's")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("improve", help="run the improvement phase on an existing solution file")
    p.add_argument("instance")
    p.add_argument("solution")
    p.add_argument("--out", default=None, help="output file (default: overwrite the input)")
    p.add_argument("--budget", type=_non_negative, default=600.0, help="seconds of simulated annealing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chains", type=_positive_int, default=1, help="parallel annealing chains")
    p.add_argument("--min-tof", type=_non_negative, default=None,
                   help="stricter minimum time of flight [days] than the instance file's")
    p.set_defaults(func=cmd_improve)
    p = sub.add_parser("evaluate", help="check a solution file and print its mission time")
    p.add_argument("instance")
    p.add_argument("solution")
    p.add_argument("--min-tof", type=_non_negative, default=None,
                   help="check against this minimum time of flight [days] instead of the file's")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("plot", help="write the figures of a solution")
    p.add_argument("instance")
    p.add_argument("solution")
    p.add_argument("--out", default="figures")
    p.set_defaults(func=cmd_plot)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, InstanceError, SolutionFileError) as exc:
        print(f"kttsp: error: {exc}", file=sys.stderr)
        sys.exit(1)
