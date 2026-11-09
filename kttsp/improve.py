"""Improvement of a constructed tour.

Three steps: spending the exception transfers that the construction left unused on the slowest
legs, simulated annealing over the visiting order with greedy re-timing of the affected part of
the tour, and a final re-timing pass with a finer search grid. Times are never part of the search
state: they are always re-derived from the order with the earliest-arrival rule, so every tour
stays feasible by construction.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from .leg_search import Leg, LegSearcher, SearchSettings
from .solution import Solution
from .transfer import leg_dv


@dataclass
class Tour:
    order: np.ndarray
    legs: list                # N-1 Leg objects

    @property
    def objective(self) -> float:
        return self.legs[-1].t_arr

    def n_exceptions(self, dv_max: float) -> int:
        return sum(1 for leg in self.legs if leg.dv > dv_max)

    def to_solution(self, instance, meta=None) -> Solution:
        exc = [k for k, leg in enumerate(self.legs) if leg.dv > instance.dv_max]
        return Solution(self.order.copy(), np.array([leg.t_dep for leg in self.legs]),
                        np.array([leg.tof for leg in self.legs]), instance=instance.name,
                        dv=np.array([leg.dv for leg in self.legs]), objective=self.objective,
                        exceptions=exc, meta=dict(meta or {}))

    @classmethod
    def from_solution(cls, sol: Solution) -> "Tour":
        legs = [Leg(int(sol.order[k]), int(sol.order[k + 1]), float(sol.t_dep[k]), float(sol.tof[k]),
                    float(sol.dv[k]), float("nan")) for k in range(sol.n - 1)]
        return cls(np.asarray(sol.order, dtype=int).copy(), legs)


class Decoder:
    """Turns a visiting order into timed legs with the earliest-arrival rule, leg by leg.
    With reuse_old=False every leg is searched again (used by the polish with a finer grid)."""

    def __init__(self, instance, eph, searcher: LegSearcher, reuse_old: bool = True):
        self.inst = instance
        self.eph = eph
        self.searcher = searcher
        self.reuse_old = reuse_old
        self.plane_dv = instance.plane_change_matrix()
        ratio = instance.a[:, None] / instance.a[None, :]
        self.cross = (ratio > searcher.s.cross_ratio) | (ratio < 1.0 / searcher.s.cross_ratio)

    def needs_exception(self, i: int, j: int) -> bool:
        """Transfers between the altitude groups or with a plane change near the limit can only
        be exception transfers; skipping the normal search saves a full scan of the horizon."""
        return bool(self.cross[i, j] or self.plane_dv[i, j] >= 0.9 * self.inst.dv_max)

    def required_after(self, order) -> np.ndarray:
        """For every leg k of the order, how many later legs change altitude group and therefore
        must be exception transfers."""
        order = np.asarray(order, dtype=int)
        cross = self.cross[order[:-1], order[1:]].astype(int)
        return np.concatenate([np.cumsum(cross[::-1])[::-1][1:], [0]])

    def _old_usable(self, old: Leg | None, i: int, j: int, t_now: float, exc_left: int) -> bool:
        """An old leg can be flown again: same pair, departs after t_now, long enough, and its
        exception (if it is one) still fits in the budget."""
        return (old is not None and old.i == i and old.j == j and old.t_dep >= t_now
                and old.tof >= self.searcher.tof_min and (old.dv <= self.inst.dv_max or exc_left > 0))

    def leg(self, i: int, j: int, t_now: float, exc_left: int, old: Leg | None = None, old_t_now: float = np.nan,
            exc_reserved: int = 0):
        """Best leg i -> j departing at or after t_now. An old leg of the same pair is kept when it
        is still valid and earlier than what the search finds, so re-timing never gets worse.
        `old_t_now` is the time the old leg was searched from (the arrival of the previous old leg);
        `exc_reserved` exceptions are kept for later legs that change altitude group."""
        inst = self.inst
        if not self.cross[i, j]:                 # this leg may only use the spare exceptions
            exc_left = exc_left - exc_reserved
        usable = self._old_usable(old, i, j, t_now, exc_left)
        if usable and self.reuse_old:
            searched_from = old_t_now if np.isfinite(old_t_now) else old.t_from
            if np.isfinite(searched_from) and searched_from <= t_now + 1e-9:
                # the old leg was found from an earlier (or equal) start and is still flyable: nothing
                # better can exist from this later start with the same search, so skip the search
                return old
        best = None
        if not self.needs_exception(i, j) or (not self.cross[i, j] and exc_left <= 0):
            best = self.searcher.earliest_arrival(i, j, t_now, inst.dv_max)
        if best is None and exc_left > 0:
            best = self.searcher.earliest_arrival(i, j, t_now, inst.dv_exception)
        if usable and (best is None or old.t_arr < best.t_arr):
            best = old
        return best

    def decode(self, order, old_legs=None, start: int = 0, stop: int | None = None, prefix=None):
        """Re-time legs start..stop-1 of `order`. `prefix` holds the legs before `start` (defaults to
        old_legs[:start]); the exception budget left after the prefix is respected. Returns the list
        of new legs or None when some leg cannot be flown."""
        inst = self.inst
        n_legs = len(order) - 1
        stop = n_legs if stop is None else stop
        prefix = (old_legs[:start] if old_legs is not None else []) if prefix is None else prefix
        exc_left = inst.n_exceptions - sum(1 for leg in prefix if leg.dv > inst.dv_max)
        t_now = prefix[-1].t_arr if prefix else float(inst.t0)
        reserved = self.required_after(order)
        out = []
        for k in range(start, stop):
            old = old_legs[k] if old_legs is not None and k < len(old_legs) else None
            old_t_now = np.nan
            if old is not None:
                old_t_now = old_legs[k - 1].t_arr if k > 0 else float(inst.t0)
            leg = self.leg(int(order[k]), int(order[k + 1]), t_now, exc_left, old, old_t_now,
                           exc_reserved=int(reserved[k]))
            if leg is None or leg.t_arr > inst.t_max:
                return None
            if leg.dv > inst.dv_max:
                exc_left -= 1
            out.append(leg)
            t_now = leg.t_arr
        return out


# ---------------------------------------------------------------------------------------------
# step 1: spare exceptions
# ---------------------------------------------------------------------------------------------

def spend_spare_exceptions(tour: Tour, dec: Decoder, n_try: int = 12, min_gain: float = 0.02, log=print) -> Tour:
    """Use the exception transfers left over by the construction on the legs where they save the
    most time; each candidate is checked by re-timing the rest of the tour."""
    inst = dec.inst
    while tour.n_exceptions(inst.dv_max) < inst.n_exceptions:
        durations = [(leg.t_arr - (tour.legs[k - 1].t_arr if k > 0 else inst.t0), k)
                     for k, leg in enumerate(tour.legs) if leg.dv <= inst.dv_max]
        durations.sort(reverse=True)
        best_tour, best_gain = None, min_gain
        for _, k in durations[:n_try]:
            t_now = tour.legs[k - 1].t_arr if k > 0 else float(inst.t0)
            cand = dec.searcher.earliest_arrival(tour.legs[k].i, tour.legs[k].j, t_now, inst.dv_exception)
            if cand is None or cand.t_arr >= tour.legs[k].t_arr - best_gain:
                continue
            suffix = dec.decode(tour.order, tour.legs, start=k + 1, prefix=tour.legs[:k] + [cand])
            if suffix is None:
                continue
            new = Tour(tour.order, tour.legs[:k] + [cand] + suffix)
            gain = tour.objective - new.objective
            if gain > best_gain and new.n_exceptions(inst.dv_max) <= inst.n_exceptions:
                best_tour, best_gain = new, gain
        if best_tour is None:
            break
        log(f"spare exception: {tour.objective:.4f} -> {best_tour.objective:.4f} d")
        tour = best_tour
    return tour


# ---------------------------------------------------------------------------------------------
# step 2: simulated annealing over the order
# ---------------------------------------------------------------------------------------------

class Annealer:
    def __init__(self, instance, dec: Decoder, rng: np.random.Generator, window: int = 5,
                 coarse_dec: Decoder | None = None, focus: float = 0.5, n_focus: int = 10):
        self.inst = instance
        self.dec = dec
        self.coarse = coarse_dec          # cheaper decoder used to screen moves before the exact one
        self.rng = rng
        self.w = window
        self.focus = focus                # share of proposals aimed at the longest legs
        self.n_focus = n_focus
        low, high = instance.groups()
        self.group = np.zeros(instance.n, dtype=int)
        self.group[low] = 1

    # ---- moves: return (new_order, first_changed_pos, last_changed_pos) or None ------------

    def _anchor(self, n: int, legs=None) -> int:
        """Position to build a move around: near one of the longest legs with probability `focus`."""
        if legs is not None and self.rng.random() < self.focus:
            dur = np.array([leg.t_arr - (legs[k - 1].t_arr if k > 0 else self.inst.t0) for k, leg in enumerate(legs)])
            worst = np.argsort(dur)[-self.n_focus:]
            k = int(self.rng.choice(worst))
            return int(np.clip(k + self.rng.integers(-2, 3), 0, n - 2))
        return int(self.rng.integers(0, n - 1))

    def _swap(self, order, legs=None):
        n = len(order)
        p = self._anchor(n, legs)
        q = p + int(self.rng.integers(1, self.w + 1))
        if q >= n or self.group[order[p]] != self.group[order[q]]:
            return None
        new = order.copy()
        new[p], new[q] = new[q], new[p]
        return new, p, q

    def _or_opt(self, order, legs=None):
        n = len(order)
        length = int(self.rng.integers(1, 4))
        if n - length < 1:                      # nothing left to move the block around
            return None
        p = min(self._anchor(n, legs), n - length)
        shift = int(self.rng.integers(-self.w, self.w + 1))
        if shift == 0:
            return None
        seg = order[p:p + length]
        if len(set(self.group[seg].tolist())) != 1:
            return None
        rest = np.concatenate([order[:p], order[p + length:]])
        q = p + shift
        if q < 0 or q > len(rest):
            return None
        neighbour = rest[q] if q < len(rest) else rest[q - 1]
        if self.group[neighbour] != self.group[seg[0]]:
            return None
        new = np.concatenate([rest[:q], seg, rest[q:]])
        lo, hi = min(p, q), max(p, q) + length - 1
        return new, lo, min(hi, n - 1)

    def _reverse(self, order, legs=None):
        n = len(order)
        p = self._anchor(n, legs)
        q = p + int(self.rng.integers(2, self.w + 2))
        if q >= n or len(set(self.group[order[p:q + 1]].tolist())) != 1:
            return None
        new = order.copy()
        new[p:q + 1] = order[p:q + 1][::-1]
        return new, p, q

    def _dip_shift(self, order, legs=None):
        """Move the tomato next to a change of altitude group to the other side of that run."""
        n = len(order)
        bounds = np.flatnonzero(self.group[order[1:]] != self.group[order[:-1]])
        if bounds.size == 0:
            return None
        b = int(self.rng.choice(bounds))        # order[b] and order[b+1] are in different groups
        g_run = self.group[order[b + 1]]
        end = b + 1
        while end + 1 < n and self.group[order[end + 1]] == g_run:
            end += 1
        if self.rng.random() < 0.5:
            # order[b] jumps behind the run
            seg = np.concatenate([order[:b], order[b + 1:end + 1], [order[b]], order[end + 1:]])
            return seg, b, end
        if end + 1 >= n:
            return None
        # order[end+1] jumps before the run
        seg = np.concatenate([order[:b + 1], [order[end + 1]], order[b + 1:end + 1], order[end + 2:]])
        return seg, b + 1, end + 1

    def propose(self, order, legs=None):
        move = self.rng.choice([self._swap, self._or_opt, self._reverse, self._dip_shift], p=[0.3, 0.35, 0.2, 0.15])
        return move(order, legs)

    # ---- evaluation ---------------------------------------------------------------------------

    def try_move(self, tour: Tour, temperature: float):
        """Apply one random move; returns the new Tour when accepted, else None."""
        res = self.propose(tour.order, tour.legs)
        if res is None:
            return None
        new_order, lo, hi = res
        n_legs = len(tour.legs)
        k0 = max(lo - 1, 0)               # first leg whose end points changed
        k1 = min(hi, n_legs - 1)          # last changed leg; position k1+1 is the first unchanged tomato
        old_arrival = tour.legs[k1].t_arr
        if self.coarse is not None:       # cheap screening of the changed window
            screen = self.coarse.decode(new_order, tour.legs, start=k0, stop=k1 + 1, prefix=tour.legs[:k0])
            if screen is None:
                return None
            delta = screen[-1].t_arr - old_arrival
            if delta > 0.0 and (temperature <= 0.0 or self.rng.random() >= math.exp(-delta / temperature)):
                return None
        window = self.dec.decode(new_order, tour.legs, start=k0, stop=k1 + 1, prefix=tour.legs[:k0])
        if window is None:
            return None
        delta = window[-1].t_arr - old_arrival
        if delta > 0.0 and (temperature <= 0.0 or self.rng.random() >= math.exp(-delta / temperature)):
            return None
        suffix = self.dec.decode(new_order, tour.legs, start=k1 + 1, prefix=tour.legs[:k0] + window)
        if suffix is None:
            return None
        new = Tour(new_order, tour.legs[:k0] + window + suffix)
        if new.n_exceptions(self.inst.dv_max) > self.inst.n_exceptions or new.objective > self.inst.t_max:
            return None
        return new


def coarse_decoder(instance, eph, base: SearchSettings | None = None) -> Decoder:
    """Decoder with a 4x coarser leg search, used only to screen annealing moves."""
    coarse = replace(base or SearchSettings(), n_dep=24, n_dep_cross=48, n_arr=48, refine_rounds=1)
    return Decoder(instance, eph, LegSearcher(eph, instance, coarse))


def local_search(tour: Tour, dec: Decoder, time_budget: float, seed: int = 0, window: int = 5,
                 t_start_frac: float = 0.01, t_end_frac: float = 1e-4, checkpoint=None, log=print,
                 coarse: Decoder | None = None) -> Tour:
    """Simulated annealing on the visiting order for `time_budget` seconds."""
    rng = np.random.default_rng(seed)
    ann = Annealer(dec.inst, dec, rng, window, coarse_dec=coarse)
    best = current = tour
    t0 = time.time()
    last_report = last_ckpt = t0
    n_tried = n_acc = 0
    T0 = t_start_frac * tour.objective
    T1 = t_end_frac * tour.objective
    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break
        frac = elapsed / time_budget
        temperature = T0 * (T1 / T0) ** frac
        n_tried += 1
        new = ann.try_move(current, temperature)
        if new is not None:
            n_acc += 1
            current = new
            if current.objective < best.objective - 1e-9:
                best = current
        now = time.time()
        if now - last_report > 60.0:
            log(f"  SA {elapsed:6.0f} s: tried {n_tried}, accepted {n_acc}, current {current.objective:.4f} d, "
                f"best {best.objective:.4f} d, T = {temperature:.4f}")
            last_report = now
        if checkpoint is not None and now - last_ckpt > 60.0:
            checkpoint(best)
            last_ckpt = now
    log(f"  SA done: tried {n_tried}, accepted {n_acc}, best {best.objective:.4f} d")
    return best


def _chain_worker(args):
    """One annealing chain in a worker process; returns the best tour as a Solution."""
    instance, solution, time_budget, seed, window, settings = args
    from .kepler import Ephemeris
    eph = Ephemeris(instance)
    dec = Decoder(instance, eph, LegSearcher(eph, instance, settings))
    tour = local_search(Tour.from_solution(solution), dec, time_budget, seed=seed, window=window,
                        log=lambda *a: None, coarse=coarse_decoder(instance, eph, settings))
    return tour.to_solution(instance)


def parallel_local_search(tour: Tour, instance, time_budget: float, chains: int, seed: int, window: int,
                          log=print, settings: SearchSettings | None = None) -> Tour:
    """Independent annealing chains with different seeds in separate processes; the best wins."""
    from concurrent.futures import ProcessPoolExecutor
    base = tour.to_solution(instance)
    settings = settings or SearchSettings()
    jobs = [(instance, base, time_budget, seed + c, window, settings) for c in range(chains)]
    with ProcessPoolExecutor(max_workers=chains) as pool:
        results = list(pool.map(_chain_worker, jobs))
    best = min(results, key=lambda s: s.t_end)
    log("  SA chains: " + ", ".join(f"{r.t_end:.4f}" for r in results) + f" d -> best {best.t_end:.4f} d")
    return Tour.from_solution(best) if best.t_end < tour.objective else tour


# ---------------------------------------------------------------------------------------------
# step 3: re-timing polish
# ---------------------------------------------------------------------------------------------

def retime_polish(tour: Tour, instance, eph, base_settings: SearchSettings | None = None, log=print) -> Tour:
    """Re-time every leg with a 2x finer search grid, keeping the old leg whenever it is better."""
    fine = replace(base_settings or SearchSettings(), n_dep=96, n_dep_cross=192, n_arr=192, refine_rounds=3)
    dec = Decoder(instance, eph, LegSearcher(eph, instance, fine), reuse_old=False)
    legs = dec.decode(tour.order, tour.legs, start=0)
    if legs is None:
        log("polish: some leg could not be re-timed, tour kept as it was")
        return tour
    new = Tour(tour.order, legs)
    old_valid = all(leg.tof >= dec.searcher.tof_min for leg in tour.legs)
    log(f"polish: {tour.objective:.4f} -> {new.objective:.4f} d")
    return new if (not old_valid or new.objective <= tour.objective) else tour


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

def improve(instance, solution: Solution, eph, searcher: LegSearcher, time_budget: float, seed: int = 0,
            window: int | None = None, checkpoint_path=None, log=print, chains: int = 1) -> Solution:
    """Full improvement pipeline on a Solution; returns a new Solution, never later than the input
    when the input already meets the search settings (a stricter minimum time of flight can force
    some legs to be re-timed)."""
    dec = Decoder(instance, eph, searcher)
    if solution.dv is None or len(solution.dv) != solution.n - 1:     # file without stored transfer costs
        dv = [leg_dv(eph, int(solution.order[k]), int(solution.order[k + 1]), float(solution.t_dep[k]),
                     float(solution.tof[k])) for k in range(solution.n - 1)]
        solution = replace(solution, dv=np.array(dv))
    tour = Tour.from_solution(solution)
    start_obj = tour.objective
    t0 = time.time()
    tour = spend_spare_exceptions(tour, dec, log=log)
    if window is None:
        window = 5 if instance.n <= 300 else 30

    def checkpoint(t: Tour):
        if checkpoint_path is not None:
            t.to_solution(instance, {**solution.meta, "checkpoint": True}).save(checkpoint_path)

    checkpoint(tour)
    remaining = time_budget - (time.time() - t0)
    annealed = remaining > 10.0
    if annealed:
        if chains > 1:
            tour = parallel_local_search(tour, instance, remaining, chains, seed, window, log=log,
                                         settings=searcher.s)
        else:
            tour = local_search(tour, dec, remaining, seed=seed, window=window, checkpoint=checkpoint, log=log,
                                coarse=coarse_decoder(instance, eph, searcher.s))
        checkpoint(tour)
        tour = spend_spare_exceptions(tour, dec, log=log)
    tour = retime_polish(tour, instance, eph, base_settings=searcher.s, log=log)
    meta = dict(solution.meta)
    meta.pop("checkpoint", None)
    # minimum time of flight used by this run (a run without one drops the earlier value)
    if searcher.s.min_tof_floor > 0.0:
        meta["min_tof_floor"] = searcher.s.min_tof_floor
    else:
        meta.pop("min_tof_floor", None)
    meta.update({"improve_seconds": round(meta.get("improve_seconds", 0) + time.time() - t0, 1),
                 "objective_before_improve": meta.get("objective_before_improve", start_obj)})
    # annealing settings, only when the annealing ran (a run that only re-times the tour keeps the earlier ones)
    if annealed:
        meta.update({"sa_seed": seed, "sa_window": window, "sa_chains": chains})
    return tour.to_solution(instance, meta)
