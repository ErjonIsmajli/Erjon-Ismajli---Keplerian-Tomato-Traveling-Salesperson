"""Lambert's problem solver: a NumPy port of the algorithm used by pykep 2.6 (Izzo, 2015).

Given two positions r1, r2, a time of flight and the gravitational parameter, the solver returns
the velocities at r1 and r2 of every Keplerian arc that connects them: the direct (0-revolution)
arc and, when the time of flight allows it, the two arcs for each number of full revolutions up
to `max_revs`. `lambert_scalar` follows pykep's control flow one problem at a time and serves as
the reference; `lambert_batch` solves many problems at once with array operations and is the one
used by the transfer search. Both reproduce pykep's formulas, constants and iteration rules so
that the set of solutions (and hence the minimum delta-v) matches the official evaluator.
"""
from __future__ import annotations

import numpy as np

_BATTIN = 0.01     # |x - 1| below this: Battin series for the time of flight
_LAGRANGE = 0.2    # |x - 1| below this (and above _BATTIN): Lagrange expression


# ---------------------------------------------------------------------------------------------
# Elementary functions of the non-dimensional Lambert variable x (all vectorised, NaN-tolerant)
# ---------------------------------------------------------------------------------------------

def _hypergeometric_f(z, tol: float = 1e-11, max_terms: int = 80):
    """Series 2F1(3, 1; 5/2; z) summed term by term until every term is below `tol`."""
    z = np.asarray(z, dtype=float)
    Sj = np.ones_like(z)
    Cj = np.ones_like(z)
    for j in range(max_terms):
        Cj = Cj * (3.0 + j) * (1.0 + j) / (2.5 + j) * z / (j + 1)
        Sj = Sj + Cj
        finite = np.abs(Cj[np.isfinite(Cj)])
        if finite.size == 0 or np.max(finite) <= tol:
            break
    return Sj


def _x2tof_lagrange(x, lam, N):
    """Time of flight from the Lagrange expression (elliptic and hyperbolic forms)."""
    a = 1.0 / (1.0 - x * x)
    # elliptic form (a > 0)
    alfa = 2.0 * np.arccos(x)
    beta = 2.0 * np.arcsin(np.sqrt(lam * lam / a))
    beta = np.where(lam < 0.0, -beta, beta)
    tof_e = (a * np.sqrt(a) * ((alfa - np.sin(alfa)) - (beta - np.sin(beta)) + 2.0 * np.pi * N)) / 2.0
    # hyperbolic form (a < 0)
    alfa_h = 2.0 * np.arccosh(x)
    beta_h = 2.0 * np.arcsinh(np.sqrt(-lam * lam / a))
    beta_h = np.where(lam < 0.0, -beta_h, beta_h)
    tof_h = (-a * np.sqrt(-a) * ((beta_h - np.sinh(beta_h)) - (alfa_h - np.sinh(alfa_h)))) / 2.0
    return np.where(a > 0.0, tof_e, tof_h)


def _x2tof(x, lam, N):
    """Non-dimensional time of flight T(x) for N revolutions, choosing pykep's three regimes."""
    x = np.asarray(x, dtype=float)
    dist = np.abs(x - 1.0)
    K = lam * lam
    E = x * x - 1.0
    rho = np.abs(E)
    z = np.sqrt(1.0 + K * E)
    # Battin series (x close to 1)
    eta = z - lam * x
    S1 = 0.5 * (1.0 - lam - x * eta)
    Q = 4.0 / 3.0 * _hypergeometric_f(S1)
    tof_battin = (eta * eta * eta * Q + 4.0 * lam * eta) / 2.0 + N * np.pi / rho ** 1.5
    # Lagrange expression (x moderately close to 1)
    tof_lagrange = _x2tof_lagrange(x, lam, N)
    # Lancaster expression (everything else)
    y = np.sqrt(rho)
    g = x * z - lam * E
    d = np.where(E < 0.0, N * np.pi + np.arccos(g), np.log(y * (z - lam * x) + g))
    tof_lancaster = (x - lam * z - d / y) / E
    return np.where(dist < _BATTIN, tof_battin, np.where(dist < _LAGRANGE, tof_lagrange, tof_lancaster))


def _dTdx(x, T, lam):
    """First, second and third derivative of T(x) at x, given T = T(x)."""
    l2 = lam * lam
    l3 = l2 * lam
    umx2 = 1.0 - x * x
    y = np.sqrt(1.0 - l2 * umx2)
    y2 = y * y
    y3 = y2 * y
    DT = 1.0 / umx2 * (3.0 * T * x - 2.0 + 2.0 * l3 * x / y)
    DDT = 1.0 / umx2 * (3.0 * T + 5.0 * x * DT + 2.0 * (1.0 - l2) * l3 / y3)
    DDDT = 1.0 / umx2 * (7.0 * x * DDT + 8.0 * DT - 6.0 * (1.0 - l2) * l2 * l3 * x / y3 / y2)
    return DT, DDT, DDDT


def _householder(T, x0, N, lam, eps, iter_max: int = 15):
    """Householder iterations solving T(x) = T, element-wise; each element stops on its own
    once its update is below `eps`, exactly like pykep's scalar loop."""
    x = np.array(x0, dtype=float, copy=True)
    done = np.zeros(x.shape, dtype=bool)
    for _ in range(iter_max):
        tof = _x2tof(x, lam, N)
        DT, DDT, DDDT = _dTdx(x, tof, lam)
        delta = tof - T
        DT2 = DT * DT
        xnew = x - delta * (DT2 - delta * DDT / 2.0) / (DT * (DT2 - delta * DDT) + DDDT * delta * delta / 6.0)
        err = np.abs(x - xnew)
        x = np.where(done, x, xnew)
        done = done | (err <= eps)
        if np.all(done):
            break
    return x


# ---------------------------------------------------------------------------------------------
# Scalar reference port (one problem, pykep's control flow)
# ---------------------------------------------------------------------------------------------

def lambert_scalar(r1, r2, tof: float, mu: float, cw: bool = False, max_revs: int = 20):
    """Solve one Lambert problem. Returns (v1, v2, x, revs): arrays of shape (B, 3), (B, 3), (B,), (B,)
    for the B = 2*Nmax+1 solutions ordered like pykep: 0 revs, then (left, right) for 1..Nmax revs.
    Raises ValueError for a non-positive time of flight or when r1 and r2 are parallel."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if tof <= 0.0:
        raise ValueError("Time of flight is negative!")
    if mu <= 0.0:
        raise ValueError("Gravity parameter is zero or negative!")
    with np.errstate(all="ignore"):
        # 1 - geometry: chord, semi-perimeter, unit vectors, lambda and the non-dimensional time T
        c = np.linalg.norm(r2 - r1)
        R1, R2 = np.linalg.norm(r1), np.linalg.norm(r2)
        s = (c + R1 + R2) / 2.0
        ir1, ir2 = r1 / R1, r2 / R2
        ih = np.cross(ir1, ir2)
        ih = ih / np.linalg.norm(ih)
        if ih[2] == 0.0 or not np.isfinite(ih[2]):
            raise ValueError("The angular momentum vector has no z component, impossible to define "
                             "automatically clock or counterclockwise")
        lambda2 = 1.0 - c / s
        lam = np.float64(np.sqrt(lambda2))
        if ih[2] < 0.0:   # transfer angle larger than 180 degrees as seen from above the z axis
            lam = -lam
            it1, it2 = np.cross(ir1, ih), np.cross(ir2, ih)
        else:
            it1, it2 = np.cross(ih, ir1), np.cross(ih, ir2)
        it1, it2 = it1 / np.linalg.norm(it1), it2 / np.linalg.norm(it2)
        if cw:            # retrograde motion
            lam = -lam
            it1, it2 = -it1, -it2
        lambda3 = lam * lambda2
        T = np.float64(np.sqrt(2.0 * mu / s / s / s) * tof)

        # 2 - maximum number of revolutions for which a solution exists
        Nmax = int(T / np.pi)
        T00 = np.arccos(lam) + lam * np.sqrt(1.0 - lambda2)
        T0 = T00 + Nmax * np.pi
        T1 = 2.0 / 3.0 * (1.0 - lambda3)
        if Nmax > 0 and T < T0:
            # Halley iterations locate the minimum time of flight of the Nmax-revolution arcs
            it = 0
            T_min = T0
            x_old = np.float64(0.0)
            x_new = np.float64(0.0)
            while True:
                DT, DDT, DDDT = _dTdx(x_old, T_min, lam)
                if DT != 0.0:
                    x_new = x_old - DT * DDT / (DDT * DDT - DT * DDDT / 2.0)
                err = abs(x_old - x_new)
                if err < 1e-13 or it > 12:
                    break
                T_min = _x2tof(x_new, lam, Nmax)
                x_old = x_new
                it += 1
            if T_min > T:
                Nmax -= 1
        Nmax = min(max_revs, Nmax)

        # 3 - one x per solution: initial guess + Householder iterations
        xs = np.empty(2 * Nmax + 1)
        revs = np.empty(2 * Nmax + 1, dtype=int)
        if T >= T00:
            x0 = -(T - T00) / (T - T00 + 4.0)
        elif T <= T1:
            x0 = T1 * (T1 - T) / (2.0 / 5.0 * (1.0 - lambda2 * lambda3) * T) + 1.0
        else:
            x0 = (T / T00) ** (0.69314718055994529 / np.log(T1 / T00)) - 1.0
        xs[0] = _householder(T, x0, 0, lam, 1e-5)
        revs[0] = 0
        for i in range(1, Nmax + 1):
            tmp = ((i * np.pi + np.pi) / (8.0 * T)) ** (2.0 / 3.0)
            xs[2 * i - 1] = _householder(T, (tmp - 1.0) / (tmp + 1.0), i, lam, 1e-8)
            tmp = ((8.0 * T) / (i * np.pi)) ** (2.0 / 3.0)
            xs[2 * i] = _householder(T, (tmp - 1.0) / (tmp + 1.0), i, lam, 1e-8)
            revs[2 * i - 1] = revs[2 * i] = i

        # 4 - terminal velocities from x
        gamma = np.sqrt(mu * s / 2.0)
        rho = (R1 - R2) / c
        sigma = np.sqrt(1.0 - rho * rho)
        y = np.sqrt(1.0 - lambda2 + lambda2 * xs * xs)
        vr1 = gamma * ((lam * y - xs) - rho * (lam * y + xs)) / R1
        vr2 = -gamma * ((lam * y - xs) + rho * (lam * y + xs)) / R2
        vt = gamma * sigma * (y + lam * xs)
        v1 = vr1[:, None] * ir1 + (vt / R1)[:, None] * it1
        v2 = vr2[:, None] * ir2 + (vt / R2)[:, None] * it2
    return v1, v2, xs, revs


# ---------------------------------------------------------------------------------------------
# Vectorised solver (many problems at once)
# ---------------------------------------------------------------------------------------------

def branch_table(max_revs: int, revs=None):
    """Numbers of revolutions and sides of the solution branches in pykep's order.
    Returns (branch_revs, branch_side) with side 0 = direct arc, -1 = left, +1 = right."""
    wanted = range(max_revs + 1) if revs is None else sorted(set(int(n) for n in revs))
    br, bs = [], []
    for n in wanted:
        if n < 0 or n > max_revs:
            continue
        if n == 0:
            br.append(0)
            bs.append(0)
        else:
            br.extend([n, n])
            bs.extend([-1, 1])
    return np.asarray(br, dtype=int), np.asarray(bs, dtype=int)


def lambert_batch(r1, r2, tof, mu: float, cw=False, max_revs: int = 20, revs=None):
    """Solve K Lambert problems at once.

    r1, r2: (K, 3) positions; tof: (K,) times of flight [s]; cw: bool or (K,) bools.
    Returns (v1, v2, branch_revs): v1 and v2 have shape (K, B, 3) where B is the number of
    solution branches in the table (0 revs, then left/right for each revolution count). Entries of
    branches that do not exist for a problem (too few revolutions possible, parallel positions,
    non-positive time of flight) are NaN. `revs` optionally restricts the revolution counts solved.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    tof = np.asarray(tof, dtype=float)
    K = tof.shape[0]
    cw = np.broadcast_to(np.asarray(cw, dtype=bool), (K,))
    branch_revs, branch_side = branch_table(max_revs, revs)
    with np.errstate(all="ignore"):
        # 1 - geometry
        c = np.linalg.norm(r2 - r1, axis=1)
        R1 = np.linalg.norm(r1, axis=1)
        R2 = np.linalg.norm(r2, axis=1)
        s = (c + R1 + R2) / 2.0
        ir1 = r1 / R1[:, None]
        ir2 = r2 / R2[:, None]
        ih = np.cross(ir1, ir2)
        ih = ih / np.linalg.norm(ih, axis=1)[:, None]
        invalid = (ih[:, 2] == 0.0) | ~np.isfinite(ih[:, 2]) | (tof <= 0.0)
        lambda2 = 1.0 - c / s
        lam = np.sqrt(lambda2)
        long_way = ih[:, 2] < 0.0
        lam = np.where(long_way, -lam, lam)
        it1 = np.where(long_way[:, None], np.cross(ir1, ih), np.cross(ih, ir1))
        it2 = np.where(long_way[:, None], np.cross(ir2, ih), np.cross(ih, ir2))
        it1 = it1 / np.linalg.norm(it1, axis=1)[:, None]
        it2 = it2 / np.linalg.norm(it2, axis=1)[:, None]
        lam = np.where(cw, -lam, lam)
        sign = np.where(cw, -1.0, 1.0)[:, None]
        it1 = it1 * sign
        it2 = it2 * sign
        lambda3 = lam * lambda2
        T = np.sqrt(2.0 * mu / s / s / s) * tof

        # 2 - maximum number of revolutions per problem (with the Halley correction of pykep)
        Nmax = np.floor(T / np.pi).astype(int)
        Nmax = np.where(invalid, 0, Nmax)
        T00 = np.arccos(lam) + lam * np.sqrt(1.0 - lambda2)
        T0 = T00 + Nmax * np.pi
        T1 = 2.0 / 3.0 * (1.0 - lambda3)
        needs_check = (Nmax > 0) & (T < T0)
        if np.any(needs_check):
            T_min = T0.copy()
            x_old = np.zeros(K)
            x_new = np.zeros(K)
            converged = ~needs_check
            for it in range(14):          # pykep: at most 14 Halley steps and 13 updates of T_min
                DT, DDT, DDDT = _dTdx(x_old, T_min, lam)
                step = np.where(DT != 0.0, x_old - DT * DDT / (DDT * DDT - DT * DDDT / 2.0), x_new)
                x_new = np.where(converged, x_new, step)
                err = np.abs(x_old - x_new)
                converged = converged | (err < 1e-13)
                if it == 13 or np.all(converged):
                    break
                update = ~converged
                T_min = np.where(update, _x2tof(x_new, lam, Nmax), T_min)
                x_old = np.where(update, x_new, x_old)
            Nmax = np.where(needs_check & (T_min > T), Nmax - 1, Nmax)
        Nmax = np.minimum(max_revs, Nmax)

        # 3 - initial guesses for every (problem, branch) and Householder iterations
        Nb = branch_revs[None, :]
        Tc = T[:, None]
        lamc = lam[:, None]
        x0_direct = np.where(
            Tc >= T00[:, None],
            -(Tc - T00[:, None]) / (Tc - T00[:, None] + 4.0),
            np.where(
                Tc <= T1[:, None],
                T1[:, None] * (T1[:, None] - Tc) / (2.0 / 5.0 * (1.0 - lambda2 * lambda3)[:, None] * Tc) + 1.0,
                (Tc / T00[:, None]) ** (0.69314718055994529 / np.log(T1 / T00)[:, None]) - 1.0,
            ),
        )
        tmp_left = ((Nb * np.pi + np.pi) / (8.0 * Tc)) ** (2.0 / 3.0)
        tmp_right = ((8.0 * Tc) / (Nb * np.pi)) ** (2.0 / 3.0)
        x0 = np.where(
            branch_side[None, :] == 0,
            x0_direct,
            np.where(branch_side[None, :] < 0, (tmp_left - 1.0) / (tmp_left + 1.0), (tmp_right - 1.0) / (tmp_right + 1.0)),
        )
        eps = np.where(branch_side[None, :] == 0, 1e-5, 1e-8)
        valid = (Nb <= Nmax[:, None]) & ~invalid[:, None]
        x0 = np.where(valid, x0, np.nan)
        xs = _householder(Tc, x0, Nb, lamc, eps)

        # 4 - terminal velocities
        gamma = np.sqrt(mu * s / 2.0)[:, None]
        rho = ((R1 - R2) / c)[:, None]
        sigma = np.sqrt(1.0 - rho * rho)
        y = np.sqrt(1.0 - lambda2[:, None] + lambda2[:, None] * xs * xs)
        vr1 = gamma * ((lamc * y - xs) - rho * (lamc * y + xs)) / R1[:, None]
        vr2 = -gamma * ((lamc * y - xs) + rho * (lamc * y + xs)) / R2[:, None]
        vt = gamma * sigma * (y + lamc * xs)
        v1 = vr1[:, :, None] * ir1[:, None, :] + (vt / R1[:, None])[:, :, None] * it1[:, None, :]
        v2 = vr2[:, :, None] * ir2[:, None, :] + (vt / R2[:, None])[:, :, None] * it2[:, None, :]
        v1 = np.where(valid[:, :, None], v1, np.nan)
        v2 = np.where(valid[:, :, None], v2, np.nan)
    return v1, v2, branch_revs
