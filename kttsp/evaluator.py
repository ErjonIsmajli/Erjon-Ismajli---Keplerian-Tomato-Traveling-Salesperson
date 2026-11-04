"""Validation of a decision vector against the official rules and computation of the objective
(total mission time). Mirrors the checks described on the challenge page."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kepler import Ephemeris
from .transfer import leg_dv


@dataclass
class LegInfo:
    k: int                # leg index (0-based)
    i_from: int
    i_to: int
    t_dep: float          # [days]
    tof: float            # [days]
    dv: float             # [m/s]
    is_exception: bool    # dv above the normal limit (allowed only for E legs)

    @property
    def t_arr(self) -> float:
        return self.t_dep + self.tof


@dataclass
class Report:
    feasible: bool
    objective: float                 # mission end time [days] (inf when the chromosome is malformed)
    violations: list = field(default_factory=list)
    legs: list = field(default_factory=list)
    n_exceptions: int = 0

    def summary(self) -> str:
        lines = [f"feasible: {self.feasible}", f"objective (mission time): {self.objective:.6f} days",
                 f"legs: {len(self.legs)}, exceptions used: {self.n_exceptions}"]
        if self.legs:
            dv = np.array([leg.dv for leg in self.legs])
            lines.append(f"delta-v per leg: max {dv.max():.3f} m/s, mean {dv.mean():.3f} m/s")
        for v in self.violations:
            lines.append(f"VIOLATION: {v}")
        return "\n".join(lines)


def split_chromosome(x, n: int):
    """Split x = [t_1..t_{N-1}, tof_1..tof_{N-1}, pi_1..pi_N] into its three parts."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size != 3 * n - 2:
        raise ValueError(f"chromosome has {x.size} values, expected 3N-2 = {3 * n - 2}")
    m = n - 1
    return x[:m].copy(), x[m:2 * m].copy(), x[2 * m:].copy()


def evaluate(instance, x, eph: Ephemeris | None = None, dv_fn=None, max_revs: int = 20,
             min_tof: float | None = None) -> Report:
    """Check every rule of the problem on the chromosome x and compute the mission time.

    dv_fn(i, j, t_dep, tof) -> delta-v can be injected to evaluate with another transfer model
    (the pykep cross-check uses this); by default the package's own `leg_dv` is used.
    min_tof replaces the minimum time of flight of the instance file when given.
    """
    n = instance.n
    if n < 2:
        return Report(False, float("inf"), ["an instance needs at least two tomatoes"])
    min_tof = instance.min_tof if min_tof is None else float(min_tof)
    try:
        x = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return Report(False, float("inf"), ["the decision vector must contain only numbers"])
    if x.size != 3 * n - 2:
        return Report(False, float("inf"), [f"chromosome has {x.size} values, expected 3N-2 = {3 * n - 2}"])
    t_dep, tof, order_f = split_chromosome(x, n)
    if not np.all(np.isfinite(order_f)):
        return Report(False, float("inf"), ["the visiting sequence contains values that are not finite numbers"])
    order = np.rint(order_f).astype(int)
    violations = []
    if np.max(np.abs(order_f - order)) > 1e-9:
        violations.append("the visiting sequence contains non-integer tomato indices")
    if sorted(order.tolist()) != list(range(n)):
        violations.append("the visiting sequence is not a permutation of 0..N-1")
    if violations:
        return Report(False, float("inf"), violations)

    if not (np.all(np.isfinite(t_dep)) and np.all(np.isfinite(tof))):
        return Report(False, float("inf"), ["departure times and times of flight must be finite numbers"])

    if eph is None:
        eph = Ephemeris(instance)
    if dv_fn is None:
        def dv_fn(i, j, t, dt):
            return leg_dv(eph, i, j, t, dt, max_revs)

    if t_dep[0] < instance.t0:
        violations.append(f"first departure at {t_dep[0]:.6f} d is before the start time {instance.t0:.6f} d")
    legs = []
    n_exc = 0
    for k in range(n - 1):
        if tof[k] < min_tof:
            violations.append(f"leg {k}: time of flight {tof[k]:.6f} d is below the minimum {min_tof:.6f} d")
        if k > 0 and t_dep[k] < t_dep[k - 1] + tof[k - 1]:
            violations.append(f"leg {k}: departure at {t_dep[k]:.6f} d is before the arrival of leg {k - 1} "
                              f"at {t_dep[k - 1] + tof[k - 1]:.6f} d")
        dv = float(dv_fn(int(order[k]), int(order[k + 1]), float(t_dep[k]), float(tof[k])))
        is_exc = not dv <= instance.dv_max
        if is_exc:
            if dv <= instance.dv_exception:
                n_exc += 1
            else:
                violations.append(f"leg {k} ({order[k]} -> {order[k + 1]}): delta-v {dv:.3f} m/s exceeds the "
                                  f"exception limit {instance.dv_exception:.1f} m/s")
        legs.append(LegInfo(k, int(order[k]), int(order[k + 1]), float(t_dep[k]), float(tof[k]), dv, is_exc))
    if n_exc > instance.n_exceptions:
        violations.append(f"{n_exc} exception transfers used, only {instance.n_exceptions} allowed")
    t_end = float(t_dep[-1] + tof[-1])
    if t_end > instance.t_max:
        violations.append(f"mission ends at {t_end:.6f} d, after the maximum time {instance.t_max:.1f} d")
    return Report(not violations, t_end, violations, legs, n_exc)
