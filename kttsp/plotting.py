"""Figures: the orbits and the tour in 3D, the delta-v landscape of one leg, the mission timeline
and the delta-v of every leg."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from .constants import DAY2SEC, R_MOON, TWO_PI  # noqa: E402
from .evaluator import evaluate  # noqa: E402
from .kepler import Ephemeris, ic2par, par2ic, solve_kepler  # noqa: E402
from .lambert import lambert_scalar  # noqa: E402
from .transfer import leg_dv_batch  # noqa: E402

GROUP_COLORS = {"low": "#d62728", "high": "#1f77b4"}


def leg_costs(instance, solution) -> tuple[np.ndarray, np.ndarray]:
    """Delta-v of every leg (stored in the solution, or computed by the evaluator) and the mask of
    the exception legs."""
    dv = solution.dv
    if dv is None or len(dv) != solution.n - 1:
        dv = np.array([leg.dv for leg in evaluate(instance, solution.to_chromosome()).legs])
    return dv, dv > instance.dv_max


def _moon(ax):
    u, v = np.mgrid[0:TWO_PI:30j, 0:np.pi:15j]
    r_km = R_MOON / 1e3
    ax.plot_surface(r_km * np.cos(u) * np.sin(v), r_km * np.sin(u) * np.sin(v), r_km * np.cos(v),
                    color="lightgray", alpha=0.6, linewidth=0)


def _finish_3d(ax, radius):
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")


def orbit_points(eph: Ephemeris, idx: int, n: int = 200) -> np.ndarray:
    t = np.linspace(0.0, eph.period[idx], n)
    r, _ = eph.state(idx, t)
    return r / 1e3


def plot_orbits(instance, indices=None, ax=None, alpha: float = 0.5):
    """All (or some) tomato orbits in 3D, coloured by altitude group."""
    eph = Ephemeris(instance)
    low, _ = instance.groups()
    low = set(low.tolist())
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
    _moon(ax)
    for idx in (range(instance.n) if indices is None else indices):
        pts = orbit_points(eph, int(idx))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=GROUP_COLORS["low" if idx in low else "high"],
                alpha=alpha, linewidth=0.6)
    _finish_3d(ax, instance.a.max() * 1.05 / 1e3)
    ax.set_title(f"{instance.name}: {instance.n} tomato orbits around the Moon")
    return ax


def transfer_arc(eph: Ephemeris, i: int, j: int, t_dep: float, tof: float, n: int = 120):
    """Points [km] along the cheapest Lambert arc of a leg (the arc the evaluator charges for)."""
    r_i, v_i = eph.state(i, t_dep)
    r_j, v_j = eph.state(j, t_dep + tof)
    best, best_v1 = np.inf, None
    for cw in (False, True):
        try:
            v1s, v2s, _, _ = lambert_scalar(r_i, r_j, tof * DAY2SEC, eph.mu, cw, 20)
        except ValueError:
            continue
        dv = np.linalg.norm(v1s - v_i, axis=1) + np.linalg.norm(v_j - v2s, axis=1)
        if not np.isfinite(dv).any():
            continue
        k = int(np.nanargmin(dv))
        if dv[k] < best:
            best, best_v1 = dv[k], v1s[k]
    if best_v1 is None:
        return np.vstack([r_i, r_j]) / 1e3
    a, e, inc, raan, argp, E0 = ic2par(r_i, best_v1, eph.mu)
    if a <= 0.0:   # hyperbolic arc (never the cheap one here): draw the chord
        return np.vstack([r_i, r_j]) / 1e3
    n_mot = np.sqrt(eph.mu / a ** 3)
    M = E0 - e * np.sin(E0) + n_mot * np.linspace(0.0, tof * DAY2SEC, n)
    E = solve_kepler(np.mod(M, TWO_PI), e)
    r, _ = par2ic(a, e, inc, raan, argp, E, eph.mu)
    return r / 1e3


def plot_tour(instance, solution, ax=None, max_legs: int | None = None):
    """The whole mission in 3D: faint orbits, blue transfer arcs, red exception transfers."""
    eph = Ephemeris(instance)
    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(projection="3d")
    plot_orbits(instance, ax=ax, alpha=0.12)
    _, exc_mask = leg_costs(instance, solution)
    n_legs = solution.n - 1 if max_legs is None else min(max_legs, solution.n - 1)
    for k in range(n_legs):
        i, j = int(solution.order[k]), int(solution.order[k + 1])
        pts = transfer_arc(eph, i, j, float(solution.t_dep[k]), float(solution.tof[k]))
        exc = bool(exc_mask[k])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="red" if exc else "navy", linewidth=1.4 if exc else 0.8,
                alpha=0.9 if exc else 0.6)
    r0, _ = eph.state(int(solution.order[0]), float(solution.t_dep[0]))
    r1, _ = eph.state(int(solution.order[-1]), float(solution.t_dep[-1] + solution.tof[-1]))
    ax.scatter(*(r0 / 1e3), color="green", s=60, label="start")
    ax.scatter(*(r1 / 1e3), color="black", s=60, label="end")
    ax.legend(loc="upper left")
    ax.set_title(f"{instance.name}: tour of {solution.n} tomatoes in {solution.t_end:.2f} days "
                 f"({int(exc_mask.sum())} exception transfers in red)")
    return ax


def plot_porkchop(instance, i: int, j: int, t_dep_range, tof_range, n: int = 150, ax=None):
    """Delta-v of the transfer i -> j over a grid of departure times and times of flight."""
    eph = Ephemeris(instance)
    t_dep = np.linspace(t_dep_range[0], t_dep_range[1], n)
    tof = np.linspace(tof_range[0], tof_range[1], n)
    dv = leg_dv_batch(eph, i, j, t_dep[:, None], tof[None, :])
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    levels = [0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 1000, 2000]
    cmap = matplotlib.colormaps["viridis_r"]
    # one colour per level, so the fine levels below the 100 m/s limit can be told apart
    cf = ax.contourf(t_dep, tof, np.clip(dv.T, 0, 2000), levels=levels, cmap=cmap, extend="max",
                     norm=BoundaryNorm(levels, cmap.N, extend="max"))
    ax.contour(t_dep, tof, dv.T, levels=[instance.dv_max], colors="white", linewidths=1.2)
    ax.contour(t_dep, tof, dv.T, levels=[instance.dv_exception], colors="orange", linewidths=0.8)
    plt.colorbar(cf, ax=ax, label="delta-v [m/s]")
    ax.set_xlabel("departure time [days]")
    ax.set_ylabel("time of flight [days]")
    ax.set_title(f"{instance.name}: transfer cost {i} -> {j} (white: {instance.dv_max:.0f} m/s, "
                 f"orange: {instance.dv_exception:.0f} m/s)")
    return ax


def plot_timeline(instance, solution, ax=None):
    """Inclination of the visited tomato against time: shows the sweep and the cluster dips."""
    low, _ = instance.groups()
    low = set(low.tolist())
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    times = np.concatenate([[solution.t_dep[0]], solution.t_dep + solution.tof])
    incs = instance.inc[solution.order]
    ax.plot(times, incs, color="gray", linewidth=0.6, zorder=1)
    colors = [GROUP_COLORS["low" if idx in low else "high"] for idx in solution.order]
    ax.scatter(times, incs, c=colors, s=14, zorder=2)
    _, exc_mask = leg_costs(instance, solution)
    for k in np.flatnonzero(exc_mask):
        ax.plot([solution.t_dep[k], solution.t_dep[k] + solution.tof[k]], [incs[k], incs[k + 1]], color="red",
                linewidth=1.5, zorder=3)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("inclination of the visited tomato [rad]")
    ax.set_title(f"{instance.name}: mission timeline ({solution.t_end:.2f} days; blue high orbits, "
                 "red low orbits, red lines exception transfers)")
    ax.grid(alpha=0.3)
    return ax


def plot_leg_dv(instance, solution, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    k = np.arange(solution.n - 1)
    dv, exc = leg_costs(instance, solution)
    ax.bar(k[~exc], dv[~exc], color="navy", width=1.0)
    # exception legs as thick lines with a marker on top, so they stay visible with a thousand legs
    ax.vlines(k[exc], ymin=1.0, ymax=dv[exc], color="red", linewidth=2.0, zorder=3)
    ax.scatter(k[exc], dv[exc], color="red", s=25, zorder=4)
    ax.axhline(instance.dv_max, color="navy", linestyle="--", linewidth=0.8)
    ax.axhline(instance.dv_exception, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("leg")
    ax.set_ylabel("delta-v [m/s]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1.0, 0.8 * float(dv.min())))
    ax.set_title(f"{instance.name}: delta-v per leg (red: the {int(exc.sum())} exception transfers; "
                 "dashed: the two limits)")
    return ax


def save_all_figures(instance, solution, outdir) -> list:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in (("tour", plot_tour), ("timeline", plot_timeline), ("leg_dv", plot_leg_dv)):
        ax = fn(instance, solution)
        p = outdir / f"{instance.name}_{name}.png"
        ax.figure.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(ax.figure)
        paths.append(p)
    ax = plot_orbits(instance)
    p = outdir / f"{instance.name}_orbits.png"
    ax.figure.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(ax.figure)
    paths.append(p)
    # porkchop of the first normal transfer between two high-altitude tomatoes
    low = set(instance.groups()[0].tolist())
    _, exc_mask = leg_costs(instance, solution)
    k = next((k for k in range(solution.n - 1)
              if not exc_mask[k] and solution.order[k] not in low and solution.order[k + 1] not in low), 0)
    i, j = int(solution.order[k]), int(solution.order[k + 1])
    T = Ephemeris(instance).period[i]
    ax = plot_porkchop(instance, i, j, (solution.t_dep[k] - 0.1 * T, solution.t_dep[k] + 1.1 * T), (0.05, 6.0 * T))
    ax.plot([solution.t_dep[k]], [solution.tof[k]], marker="*", color="red", markersize=12)
    p = outdir / f"{instance.name}_porkchop.png"
    ax.figure.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(ax.figure)
    paths.append(p)
    return paths
