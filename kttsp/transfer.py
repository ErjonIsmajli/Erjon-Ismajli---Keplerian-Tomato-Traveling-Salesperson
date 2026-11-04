"""Delta-v of a transfer between two tomatoes, defined exactly like the official `compute_transfer`:
the cheapest Lambert arc over both directions of motion and all revolution counts up to 20, where the
cost is |v1 - v_departure| + |v_arrival - v2| (two impulsive burns)."""
from __future__ import annotations

import numpy as np

from .constants import DAY2SEC
from .lambert import lambert_batch, lambert_scalar


def leg_dv(eph, i: int, j: int, t_dep: float, tof: float, max_revs: int = 20) -> float:
    """Delta-v [m/s] of the transfer from tomato i (departing at t_dep days) to tomato j
    (arriving tof days later); inf when no Lambert arc exists."""
    if not tof > 0.0:
        return float("inf")
    r_i, v_i = eph.state(i, t_dep)
    r_j, v_j = eph.state(j, t_dep + tof)
    best = float("inf")
    for cw in (False, True):
        try:
            v1, v2, _, _ = lambert_scalar(r_i, r_j, tof * DAY2SEC, eph.mu, cw, max_revs)
        except ValueError:
            continue
        dv = np.linalg.norm(v1 - v_i, axis=1) + np.linalg.norm(v_j - v2, axis=1)
        dv = dv[np.isfinite(dv)]
        if dv.size and dv.min() < best:
            best = float(dv.min())
    return best


def _cheapest_branch(v1, v2, v_i, v_j) -> np.ndarray:
    """Minimum over the branch axis of the two-burn cost; non-existing branches count as inf."""
    dv = np.linalg.norm(v1 - v_i[:, None, :], axis=2) + np.linalg.norm(v_j[:, None, :] - v2, axis=2)
    dv = np.where(np.isfinite(dv), dv, np.inf)
    return dv.min(axis=1)


def leg_dv_batch(eph, i, j, t_dep, tof, max_revs: int = 20, revs=None, both_cw: bool = True,
                 guard_angle: float = 0.0) -> np.ndarray:
    """Delta-v [m/s] of many transfers at once. `i`, `j` are tomato indices (scalars or arrays),
    `t_dep` and `tof` arrays of days; all broadcast to a common shape.

    With both_cw=False only the direction of motion of the departing tomato is tried (both are
    tried where the two tomatoes orbit in opposite senses). That is a subset of the official
    search, so the result is an upper bound of the official cost: anything feasible here is
    feasible officially. `revs` restricts the revolution counts for the same reason. With
    guard_angle > 0, geometries whose transfer angle is within that many radians of 0 or pi
    (undefined transfer plane) are reported as inf.
    """
    t_dep = np.asarray(t_dep, dtype=float)
    tof = np.asarray(tof, dtype=float)
    shape = np.broadcast(t_dep, tof, np.asarray(i), np.asarray(j)).shape
    t_dep = np.broadcast_to(t_dep, shape).ravel()
    tof = np.broadcast_to(tof, shape).ravel()
    i = np.broadcast_to(np.asarray(i, dtype=int), shape).ravel()
    j = np.broadcast_to(np.asarray(j, dtype=int), shape).ravel()
    r_i, v_i = eph.state(i, t_dep)
    r_j, v_j = eph.state(j, t_dep + tof)
    tof_s = tof * DAY2SEC
    if both_cw:
        best = np.full(tof.shape, np.inf)
        for cw in (False, True):
            v1, v2, _ = lambert_batch(r_i, r_j, tof_s, eph.mu, cw, max_revs, revs)
            best = np.minimum(best, _cheapest_branch(v1, v2, v_i, v_j))
    else:
        cw = ~eph.prograde[i]
        v1, v2, _ = lambert_batch(r_i, r_j, tof_s, eph.mu, cw, max_revs, revs)
        best = _cheapest_branch(v1, v2, v_i, v_j)
        mixed = eph.prograde[i] != eph.prograde[j]
        if np.any(mixed):
            v1, v2, _ = lambert_batch(r_i[mixed], r_j[mixed], tof_s[mixed], eph.mu, ~cw[mixed], max_revs, revs)
            best[mixed] = np.minimum(best[mixed], _cheapest_branch(v1, v2, v_i[mixed], v_j[mixed]))
    best = np.where(tof > 0.0, best, np.inf)
    if guard_angle > 0.0:
        cosang = np.sum(r_i * r_j, axis=-1) / (np.linalg.norm(r_i, axis=-1) * np.linalg.norm(r_j, axis=-1))
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        best = np.where(np.minimum(ang, np.pi - ang) < guard_angle, np.inf, best)
    return best.reshape(shape)


def transfer_angle(eph, i, j, t_dep, tof) -> np.ndarray:
    """Angle [rad] between the departure and the arrival position vectors (0 or pi = degenerate
    Lambert geometry where the transfer plane is undefined)."""
    t_dep = np.asarray(t_dep, dtype=float)
    tof = np.asarray(tof, dtype=float)
    r_i, _ = eph.state(i, t_dep)
    r_j, _ = eph.state(j, t_dep + tof)
    cosang = np.sum(r_i * r_j, axis=-1) / (np.linalg.norm(r_i, axis=-1) * np.linalg.norm(r_j, axis=-1))
    return np.arccos(np.clip(cosang, -1.0, 1.0))
