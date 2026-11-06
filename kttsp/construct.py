"""Construction of a complete tour.

The instance has a clear structure: two altitude groups, the high group forming one chain of
inclinations that normal transfers can follow, the low group forming a few inclination clusters
that can only be entered and left with exception transfers. A macro plan fixes the order of these
blocks (and where the exceptions are spent); inside a block a greedy rule picks, among the next
few tomatoes of the sweep, the one that can be reached earliest.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .evaluator import evaluate
from .kepler import Ephemeris
from .leg_search import LegSearcher, SearchSettings
from .solution import Solution


@dataclass
class Block:
    name: str
    indices: np.ndarray        # tomatoes of the block
    sweep: int                 # +1 / -1: visit in increasing / decreasing inclination, 0: nearest first
    exception_in: bool         # entering this block needs an exception transfer


@dataclass
class Structure:
    low: np.ndarray
    high: np.ndarray
    low_clusters: list         # list of index arrays, sorted by their median inclination
    high_chain: np.ndarray     # high tomatoes sorted by inclination
    cluster_inc: np.ndarray    # median inclination of every low cluster


@dataclass
class ConstructSettings:
    window: int = 6                # how many upcoming tomatoes of the sweep are candidates
    n_candidates: int = 4          # how many of them get a full leg search (after pre-ranking)
    entry_candidates: int = 8      # candidates considered when entering a new block
    plane_fraction: float = 0.85   # plane change of a candidate must stay below this fraction of the limit
    straggler_fraction: float = 0.5  # and the tomato furthest behind must stay within this fraction
    lookahead: int = 3             # candidates judged by their best successor (0 = pure earliest arrival)
    lookahead_successors: int = 2  # successors searched per candidate in the lookahead
    max_starts: int = 3            # start tomatoes tried in the first block
    both_directions: bool = True
    start_in_high: bool = True     # also try plans that start on the high chain
    workers: int = 1               # processes used to build the (plan, start) combinations
    verbose: bool = False


def analyse(instance) -> Structure:
    """Altitude groups and inclination clusters of an instance."""
    low, high = instance.groups()
    v_low = float(np.mean(instance.v_circ()[low])) if low.size else 1.0
    gap = instance.max_plane_change(instance.dv_exception, v_low)
    clusters = instance.clusters(low, gap) if low.size else []
    med = np.array([float(np.median(instance.inc[c])) for c in clusters])
    order = np.argsort(med) if med.size else np.array([], dtype=int)
    clusters = [clusters[k] for k in order]
    chain = high[np.argsort(instance.inc[high])]
    return Structure(low, high, clusters, chain, med[order] if med.size else med)


def make_plans(instance, structure: Structure, settings: ConstructSettings) -> list:
    """Macro plans: the low clusters in sweep order, separated by the pieces of the high chain that
    lie between their inclinations; the piece before the last cluster takes the whole rest of the
    chain so that the mission can end inside that cluster. Plans exist for both sweep directions
    and for starting either inside the first cluster or on the high chain; plans needing more
    exception transfers than allowed are dropped."""
    inc = instance.inc
    chain = structure.high_chain
    plans = []
    directions = [1, -1] if settings.both_directions else [1]
    for direction in directions:
        cl = list(zip(structure.low_clusters, structure.cluster_inc))
        if direction == -1:
            cl = cl[::-1]
        if not cl:
            plans.append([Block("high", chain, direction, False)])
            continue
        for start_high in ([False, True] if settings.start_in_high else [False]):
            blocks = []
            remaining = chain.copy()

            def before(cut, rem):
                return rem[(direction * inc[rem]) < (direction * cut)]

            if start_high:
                part = before(cl[0][1], remaining)
                if part.size == 0:
                    continue
                blocks.append(Block("high-0", part, direction, False))
                remaining = np.setdiff1d(remaining, part)
                blocks.append(Block("low-0", cl[0][0], 0, True))
            else:
                blocks.append(Block("low-0", cl[0][0], 0, False))
            for t in range(1, len(cl)):
                part = before(cl[t][1], remaining) if t < len(cl) - 1 else remaining
                if part.size:
                    blocks.append(Block(f"high-{t}", part, direction, True))
                    remaining = np.setdiff1d(remaining, part)
                blocks.append(Block(f"low-{t}", cl[t][0], 0, True))
            if remaining.size:
                blocks.append(Block(f"high-{len(cl)}", remaining, direction, True))
            covered = np.concatenate([b.indices for b in blocks])
            assert sorted(covered.tolist()) == list(range(instance.n)), "plan does not cover every tomato"
            if sum(1 for b in blocks if b.exception_in) <= instance.n_exceptions:
                plans.append(blocks)
    return plans


class GreedyBuilder:
    """Builds a tour for one macro plan with the earliest-arrival rule."""

    def __init__(self, instance, eph: Ephemeris, searcher: LegSearcher, settings: ConstructSettings):
        self.inst = instance
        self.eph = eph
        self.searcher = searcher
        self.s = settings
        self.plane_dv = instance.plane_change_matrix()

    # ---- candidate ranking without any Lambert call -----------------------------------

    def _estimate_arrival(self, i: int, cands: np.ndarray, t_now: float, dv_limit: float) -> np.ndarray:
        """Rough arrival time per candidate from the phase difference and the plane-change cost:
        a transfer of N revolutions can correct a phase error dtheta at a cost v*dtheta/(3*pi*N)."""
        eph = self.eph
        v = eph.v_circ[i]
        T = eph.period[i]
        budget = dv_limit - self.searcher.s.dv_margin - self.plane_dv[i, cands]
        phase_i = eph.phase(i, t_now)
        phase_c = eph.phase(cands, t_now)
        drift = (eph.n[cands] - eph.n[i]) * 86400.0          # relative phase drift [rad/day]
        est = np.full(cands.shape, np.inf)
        for N in range(0, 21):
            dtheta = np.mod(phase_c - phase_i + drift * N * T, 2.0 * np.pi)
            dtheta = np.minimum(dtheta, 2.0 * np.pi - dtheta)
            cost = np.where(N == 0, v * dtheta, v * dtheta / (3.0 * np.pi * max(N, 1)))
            feasible = (budget > 0.0) & (cost <= budget) & ~np.isfinite(est)
            est = np.where(feasible, t_now + (N + 0.5) * T, est)
        return est

    def _candidate_legs(self, i: int, cands: np.ndarray, t_now: float, dv_limit: float, n_eval: int) -> list:
        """Full leg search for the most promising candidates (by the cheap estimate); more candidates
        are tried only while nothing feasible has been found. Returns the legs sorted by arrival."""
        if cands.size == 0:
            return []
        est = self._estimate_arrival(i, cands, t_now, dv_limit)
        ranked = cands[np.argsort(est, kind="stable")]
        legs = []
        for k, c in enumerate(ranked):
            if k >= n_eval and legs:
                break
            leg = self.searcher.earliest_arrival(i, int(c), t_now, dv_limit)
            if leg is not None:
                legs.append(leg)
        legs.sort(key=lambda leg: leg.t_arr)
        return legs

    def _best_leg(self, i: int, cands: np.ndarray, t_now: float, dv_limit: float, n_eval: int):
        legs = self._candidate_legs(i, cands, t_now, dv_limit, n_eval)
        return legs[0] if legs else None

    def _choose_with_lookahead(self, legs: list, block: Block, visited: np.ndarray, sweep: int):
        """Among the earliest legs, prefer the one whose best successor inside the block arrives first."""
        inst = self.inst
        if self.s.lookahead <= 1 or len(legs) <= 1:
            return legs[0]
        best, best_score = None, np.inf
        for leg in legs[: self.s.lookahead]:
            visited[leg.j] = True
            succ = self._block_candidates(block, visited, leg.j, inst.dv_max, sweep)
            visited[leg.j] = False
            if succ.size == 0:                       # the block ends with this tomato
                score = leg.t_arr
            else:
                nxt = self._candidate_legs(leg.j, succ, leg.t_arr, inst.dv_max, self.s.lookahead_successors)
                score = nxt[0].t_arr if nxt else np.inf
            if score < best_score - 1e-9 or (abs(score - best_score) <= 1e-9 and leg.t_arr < best.t_arr):
                best, best_score = leg, score
        return best if best is not None else legs[0]

    # ---- building -----------------------------------------------------------------------

    def _sorted_remaining(self, block: Block, visited: np.ndarray, i: int, sweep: int) -> np.ndarray:
        rem = block.indices[~visited[block.indices]]
        inc = self.inst.inc
        if rem.size == 0:
            return rem
        if sweep == 0:
            return rem[np.argsort(np.abs(inc[rem] - inc[i]))]
        return rem[np.argsort(sweep * inc[rem])]

    def _block_candidates(self, block: Block, visited: np.ndarray, i: int, limit: float, sweep: int,
                          strict: bool = True) -> np.ndarray:
        """Unvisited tomatoes of the block that are next in the sweep and within plane-change reach.
        With `strict`, candidates must also stay close (in plane change) to the tomato furthest behind
        in the sweep, so that the sweep never leaves a straggler that is too expensive to collect."""
        rem = self._sorted_remaining(block, visited, i, sweep)
        if rem.size == 0:
            return rem
        behind = rem[0]
        cands = rem[: self.s.window]
        reach = self.plane_dv[i, cands] <= self.s.plane_fraction * limit
        if strict and sweep != 0:
            reach &= (cands == behind) | (self.plane_dv[cands, behind] <= self.s.straggler_fraction * limit)
        return cands[reach]

    def _entry_candidates(self, block: Block, visited: np.ndarray, i: int, sweep: int) -> np.ndarray:
        """Tomatoes through which a new block is entered: the first ones of its sweep that keep the
        very first one within easy reach, or the closest ones in inclination for a cluster."""
        rem = self._sorted_remaining(block, visited, i, sweep)
        cands = rem[: self.s.entry_candidates]
        if sweep != 0 and cands.size:
            keep = self.plane_dv[cands, rem[0]] <= self.s.straggler_fraction * self.inst.dv_max
            cands = cands[keep] if keep.any() else cands[:1]
        return cands

    @staticmethod
    def _spare_exceptions(plan: list, b: int, exc_left: int) -> int:
        """Exceptions that can be spent now without starving the blocks after block b that must be
        entered with an exception transfer."""
        return exc_left - sum(1 for blk in plan[b + 1:] if blk.exception_in)

    def _cluster_direction(self, block: Block, entry: int) -> int:
        """Sweep direction inside a cluster: away from the end through which it was entered."""
        inc = self.inst.inc
        return 1 if inc[entry] <= np.median(inc[block.indices]) else -1

    def build(self, plan: list, start: int):
        """Greedy tour for a plan starting at tomato `start` (a member of the first block).
        Returns (Solution, log) or (None, reason)."""
        inst = self.inst
        n = inst.n
        visited = np.zeros(n, dtype=bool)
        order = [int(start)]
        visited[start] = True
        legs = []
        exceptions = []
        exc_left = inst.n_exceptions
        t_now = float(inst.t0)
        current = int(start)
        for b, block in enumerate(plan):
            first_in_block = True
            sweep = block.sweep if block.sweep != 0 else self._cluster_direction(block, current) if b == 0 else 0
            while not np.all(visited[block.indices]):
                entering = first_in_block and b > 0
                if entering:
                    rem = self._entry_candidates(block, visited, current, block.sweep)
                    limit = inst.dv_exception if block.exception_in else inst.dv_max
                    leg = self._best_leg(current, rem, t_now, limit, len(rem))
                    if leg is None and not block.exception_in and self._spare_exceptions(plan, b, exc_left) > 0:
                        leg = self._best_leg(current, rem, t_now, inst.dv_exception, len(rem))
                    if leg is not None and block.sweep == 0:
                        sweep = self._cluster_direction(block, leg.j)
                else:
                    cands = self._block_candidates(block, visited, current, inst.dv_max, sweep)
                    legs_found = self._candidate_legs(current, cands, t_now, inst.dv_max, self.s.n_candidates)
                    leg = self._choose_with_lookahead(legs_found, block, visited, sweep) if legs_found else None
                    if leg is None:   # relax the reach filters before giving up
                        cands = self._block_candidates(block, visited, current, inst.dv_max, sweep, strict=False)
                        leg = self._best_leg(current, cands, t_now, inst.dv_max, self.s.n_candidates)
                    if leg is None:
                        rem = self._sorted_remaining(block, visited, current, sweep)
                        leg = self._best_leg(current, rem[: self.s.window], t_now, inst.dv_max, self.s.n_candidates)
                    if leg is None and self._spare_exceptions(plan, b, exc_left) > 0:   # spare exception
                        rem = self._sorted_remaining(block, visited, current, sweep)
                        leg = self._best_leg(current, rem[: self.s.window], t_now, inst.dv_exception,
                                             self.s.n_candidates)
                if leg is None:
                    return None, f"stuck at tomato {current} in block {block.name} at t = {t_now:.2f} d"
                if leg.dv > inst.dv_max:
                    if exc_left == 0:
                        return None, f"out of exceptions at tomato {current} in block {block.name}"
                    exc_left -= 1
                    exceptions.append(len(legs))
                legs.append(leg)
                order.append(leg.j)
                visited[leg.j] = True
                current = leg.j
                t_now = leg.t_arr
                first_in_block = False
                if self.s.verbose:
                    print(f"  leg {len(legs):4d}: {leg.i:4d} -> {leg.j:4d}  dep {leg.t_dep:9.4f}  "
                          f"tof {leg.tof:8.4f}  arr {leg.t_arr:9.4f}  dv {leg.dv:7.2f}"
                          + ("  EXC" if leg.dv > inst.dv_max else ""))
        if len(order) != n:
            return None, "plan did not cover every tomato"
        sol = Solution(np.array(order), np.array([leg.t_dep for leg in legs]), np.array([leg.tof for leg in legs]),
                       instance=inst.name, dv=np.array([leg.dv for leg in legs]), objective=legs[-1].t_arr,
                       exceptions=exceptions)
        return sol, "ok"


def _start_tomatoes(instance, plan, settings: ConstructSettings, seed: int = 0) -> list:
    """Start tomatoes tried for a plan: the ends of the first block's inclination range first."""
    first = plan[0]
    inc = instance.inc[first.indices]
    rng = np.random.default_rng(seed)
    if first.sweep == 0:
        srt = first.indices[np.argsort(inc)]
        starts = np.concatenate([srt[:1], srt[-1:], rng.permutation(srt[1:-1])])
    else:
        starts = first.indices[np.argsort(first.sweep * inc)]
    return list(dict.fromkeys(int(v) for v in starts))[: settings.max_starts]


def _build_job(args):
    """One (plan, start) construction; runs in a worker process."""
    instance, settings, search_settings, plan_idx, start = args
    eph = Ephemeris(instance)
    searcher = LegSearcher(eph, instance, search_settings)
    plans = make_plans(instance, analyse(instance), settings)
    builder = GreedyBuilder(instance, eph, searcher, settings)
    t0 = time.time()
    sol, msg = builder.build(plans[plan_idx], start)
    return plan_idx, start, sol, msg, time.time() - t0, searcher.n_searches, searcher.n_grid_points


def construct(instance, eph: Ephemeris | None = None, searcher: LegSearcher | None = None,
              settings: ConstructSettings | None = None, search_settings: SearchSettings | None = None,
              log=print):
    """Try every macro plan (and a few start tomatoes each) and return the best feasible Solution."""
    settings = settings or ConstructSettings()
    search_settings = search_settings or (searcher.s if searcher is not None else SearchSettings())
    eph = eph or Ephemeris(instance)
    plans = make_plans(instance, analyse(instance), settings)
    jobs = [(instance, settings, search_settings, p, start)
            for p, plan in enumerate(plans) for start in _start_tomatoes(instance, plan, settings)]
    if settings.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(settings.workers, len(jobs))) as pool:
            results = list(pool.map(_build_job, jobs))
    else:
        searcher = searcher or LegSearcher(eph, instance, search_settings)
        builder = GreedyBuilder(instance, eph, searcher, settings)
        results = []
        for _, _, _, p, start in jobs:
            t0, s0, g0 = time.time(), searcher.n_searches, searcher.n_grid_points
            sol, msg = builder.build(plans[p], start)
            results.append((p, start, sol, msg, time.time() - t0, searcher.n_searches - s0,
                            searcher.n_grid_points - g0))
    best = None
    n_searches = sum(r[5] for r in results)
    n_grid = sum(r[6] for r in results)
    for p, start, sol, msg, secs, _, _ in results:
        desc = " > ".join(f"{b.name}[{b.indices.size}]" for b in plans[p])
        if sol is None:
            log(f"plan {p} start {start}: failed ({msg}) [{secs:.0f} s]  {desc}")
            continue
        log(f"plan {p} start {start}: {sol.objective:.4f} d, {len(sol.exceptions)} exceptions [{secs:.0f} s]  {desc}")
        if best is None or sol.objective < best.objective:
            best = sol
            best.meta.update({"plan": p, "start": start, "blocks": desc})
    if best is not None:
        best.meta.update({"leg_searches": n_searches, "grid_points": n_grid})
        report = evaluate(instance, best.to_chromosome(), eph)
        best.objective = report.objective
        best.meta["feasible"] = report.feasible
        if not report.feasible:
            log("WARNING: constructed tour fails the evaluator: " + "; ".join(report.violations))
    return best
