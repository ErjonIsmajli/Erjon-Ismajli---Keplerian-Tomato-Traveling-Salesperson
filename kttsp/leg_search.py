"""Search of the earliest possible arrival at a target tomato for one leg under a delta-v limit.

The delta-v of a leg depends on the departure time and on the time of flight in a very oscillating
way (every revolution of the transfer orbit opens a new window), so the search is a grid scan:
departure times over one period of the departing orbit, arrival times in chunks of one period,
scanned chunk by chunk until a feasible pair appears. The first feasible chunk is refined locally
and the result is confirmed with the exact (official) transfer model with a safety margin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transfer import leg_dv, leg_dv_batch


@dataclass(frozen=True)
class Leg:
    i: int
    j: int
    t_dep: float   # [days]
    tof: float     # [days]
    dv: float      # [m/s], exact value from the official transfer model
    t_from: float = float("nan")   # earliest departure the search started from

    @property
    def t_arr(self) -> float:
        return self.t_dep + self.tof


@dataclass
class SearchSettings:
    dv_margin: float = 0.5        # the solver stays this far [m/s] below every delta-v limit
    tof_margin: float = 1e-6      # and this far [days] above the minimum time of flight
    n_dep: int = 48               # departure grid points over one period of the departing orbit
    n_dep_cross: int = 96         # same, for transfers between the two altitude groups
    n_arr: int = 96               # arrival grid points per period of the faster of the two orbits
    horizon_days: float = 45.0    # arrival times scanned at most this far from the earliest departure
    guard_angle: float = 1e-4     # [rad] grid points with a (nearly) collinear geometry are skipped
    refine_rounds: int = 2
    refine_factor: int = 8
    cross_ratio: float = 1.2      # semi-major axes differing by more than this: cross-group transfer
    min_tof_floor: float = 0.0    # [days] stricter minimum time of flight than the instance file's (0 = file value)


class LegSearcher:
    """Finds legs with `earliest_arrival`; results are cached per (i, j, limit, time bucket)."""

    def __init__(self, eph, instance, settings: SearchSettings | None = None):
        self.eph = eph
        self.inst = instance
        self.s = settings or SearchSettings()
        self.cache: dict = {}
        self.n_searches = 0
        self.n_grid_points = 0

    # ---- public API -----------------------------------------------------------------

    @property
    def tof_min(self) -> float:
        """Smallest time of flight [days] the search accepts: the file's (or the stricter) minimum plus the margin."""
        return max(self.inst.min_tof, self.s.min_tof_floor) + self.s.tof_margin

    def earliest_arrival(self, i: int, j: int, t_now: float, dv_limit: float, horizon_days: float | None = None):
        """Leg from i to j departing at or after t_now with delta-v <= dv_limit (minus the margin)
        and the smallest arrival time the grid search can find; None when nothing is found within
        the horizon. A cached result is reused only when it was searched from an earlier (or the same)
        time and still departs in the future: a later start can never arrive earlier."""
        bucket = int(t_now // (self.eph.period[i] / 10.0))
        key = (int(i), int(j), round(float(dv_limit), 3), bucket, horizon_days)
        if key in self.cache:
            t_from, hit = self.cache[key]
            if t_from <= t_now and (hit is None or hit.t_dep >= t_now):
                return hit
        leg = self._search(int(i), int(j), float(t_now), float(dv_limit), horizon_days)
        self.cache[key] = (float(t_now), leg)
        return leg

    def exact_dv(self, i: int, j: int, t_dep: float, tof: float) -> float:
        return leg_dv(self.eph, i, j, t_dep, tof)

    # ---- internals ------------------------------------------------------------------

    def _is_cross(self, i: int, j: int) -> bool:
        ratio = self.eph.a[i] / self.eph.a[j]
        return ratio > self.s.cross_ratio or ratio < 1.0 / self.s.cross_ratio

    def _transfer_period(self, i: int, j: int) -> float:
        """Period [days] of the orbit the cheap transfers roughly follow: the departing orbit for
        transfers inside a group, the Hohmann-like ellipse between the two radii otherwise."""
        a_t = 0.5 * (self.eph.a[i] + self.eph.a[j])
        return float(2.0 * np.pi * np.sqrt(a_t ** 3 / self.eph.mu) / 86400.0)

    def _revs_for(self, tof_lo: float, tof_hi: float, T_t: float):
        """Revolution counts worth solving for times of flight in [tof_lo, tof_hi]."""
        lo = max(0, int(np.floor(tof_lo / T_t)) - 1)
        hi = min(20, int(np.floor(tof_hi / T_t)) + 1)
        return range(lo, hi + 1)

    def _grid_dv(self, i, j, t_dep, tof, revs):
        self.n_grid_points += int(np.size(tof))
        return leg_dv_batch(self.eph, i, j, t_dep, tof, revs=revs, both_cw=False, guard_angle=self.s.guard_angle)

    def _search(self, i, j, t_now, dv_limit, horizon_days):
        """Chunk-by-chunk grid scan; returns the first confirmed leg (earliest arrival) or None."""
        s = self.s
        self.n_searches += 1
        cross = self._is_cross(i, j)
        T_i = float(self.eph.period[i])
        T_ref = float(min(self.eph.period[i], self.eph.period[j]))
        T_t = self._transfer_period(i, j)
        limit = dv_limit - s.dv_margin
        tof_min = self.tof_min
        n_dep = s.n_dep_cross if cross else s.n_dep
        d_dep = T_i / n_dep
        d_arr = T_ref / s.n_arr
        width = T_i                                   # one chunk of arrival times
        horizon = min(horizon_days or s.horizon_days, 20.5 * T_t, self.inst.t_max - t_now)
        t_dep_grid = t_now + d_dep * np.arange(n_dep)
        n_arr = int(np.ceil(width / d_arr))
        n_chunks = int(np.ceil(horizon / width))
        for m in range(max(1, n_chunks)):
            a0 = t_now + m * width
            t_arr_grid = a0 + d_arr * np.arange(n_arr)
            TD, TA = np.meshgrid(t_dep_grid, t_arr_grid, indexing="ij")
            TOF = TA - TD
            ok = (TOF >= tof_min) & (TA <= t_now + horizon)
            if not ok.any():
                continue
            td, tof = TD[ok], TOF[ok]
            revs = self._revs_for(float(tof.min()), float(tof.max()), T_t)
            dv = self._grid_dv(i, j, td, tof, revs)
            feas = dv <= limit
            if not feas.any():
                continue
            k = int(np.argmin(np.where(feas, td + tof, np.inf)))
            cand = self._refine(i, j, t_now, limit, tof_min, (float(td[k]), float(tof[k])), d_dep, d_arr, revs)
            dv_exact = self.exact_dv(i, j, cand[0], cand[1])
            if dv_exact <= limit:
                return Leg(i, j, cand[0], cand[1], dv_exact, t_now)
            dv_exact = self.exact_dv(i, j, float(td[k]), float(tof[k]))   # fall back to the grid point
            if dv_exact <= limit:
                return Leg(i, j, float(td[k]), float(tof[k]), dv_exact, t_now)
        return None

    def _refine(self, i, j, t_now, limit, tof_min, cand, d_dep, d_arr, revs):
        """Zoom in around a feasible grid point looking for an earlier feasible arrival."""
        f = self.s.refine_factor
        td0, tof0 = cand
        for _ in range(self.s.refine_rounds):
            ta0 = td0 + tof0
            td = td0 + d_dep * np.linspace(-1.0, 1.0, 2 * f + 1)
            ta = ta0 + d_arr * np.linspace(-1.0, 0.0, f + 1)
            TD, TA = np.meshgrid(td, ta, indexing="ij")
            TOF = TA - TD
            ok = (TD >= t_now) & (TOF >= tof_min)
            if not ok.any():
                break
            td_f, tof_f = TD[ok], TOF[ok]
            dv = self._grid_dv(i, j, td_f, tof_f, revs)
            feas = dv <= limit
            if feas.any():
                k = int(np.argmin(np.where(feas, td_f + tof_f, np.inf)))
                td0, tof0 = float(td_f[k]), float(tof_f[k])
            d_dep /= f
            d_arr /= f
        return td0, tof0
