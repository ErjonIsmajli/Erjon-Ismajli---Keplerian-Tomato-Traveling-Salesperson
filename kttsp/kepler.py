"""Two-body (Keplerian) orbit propagation with the same conventions as pykep 2.x.

Angles are in radians, distances in metres, velocities in m/s and the epoch of an ephemeris in days.
"""
from __future__ import annotations

import numpy as np

from .constants import DAY2SEC, MU_MOON, TWO_PI


def _kepler_bracketed(M, e, tol: float, max_iter: int = 200) -> np.ndarray:
    """Newton's method kept inside the interval [M - e, M + e] that always contains the root
    (|E - M| = e*|sin E| <= e): whenever a Newton step would leave the current bracket, a bisection
    step is taken instead, so it converges for every eccentricity below 1."""
    E = M + e * np.cos(M)
    lo, hi = M - e, M + e
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        lo = np.where(f < 0.0, E, lo)              # the root lies above E
        hi = np.where(f > 0.0, E, hi)              # the root lies below E
        E_new = E - f / (1.0 - e * np.cos(E))
        outside = (f != 0.0) & ~((E_new > lo) & (E_new < hi))
        E_new = np.where(outside, 0.5 * (lo + hi), E_new)
        converged = np.max(np.abs(E_new - E)) < tol
        E = E_new
        if converged:
            break
    return E


def solve_kepler(M, e, tol: float = 1e-13, max_iter: int = 100) -> np.ndarray:
    """Solve Kepler's equation E - e*sin(E) = M for the eccentric anomaly E, element-wise on arrays.

    Newton's method from pykep's starting guess E = M + e*cos(M), which is fast and exact for the
    nearly circular tomato orbits. Plain Newton steps can diverge on very eccentric orbits, so the
    elements that did not converge are solved again with a bracketed Newton method."""
    M, e = np.broadcast_arrays(np.asarray(M, dtype=float), np.asarray(e, dtype=float))
    E = M + e * np.cos(M)
    if E.size == 0:
        return E
    for _ in range(max_iter):
        step = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - step
        if np.max(np.abs(step)) < tol:
            break
    bad = ~(np.abs(E - e * np.sin(E) - M) < 1e-10)
    if np.any(bad):
        E = np.array(E, dtype=float, copy=True)
        E[bad] = _kepler_bracketed(M[bad], e[bad], tol)
    return E


def perifocal_rotation(inc, raan, argp) -> np.ndarray:
    """Rotation matrices from the perifocal frame to the inertial frame, R = Rz(raan) Rx(inc) Rz(argp),
    written out exactly like pykep's par2ic so that negative inclinations behave identically."""
    inc, raan, argp = np.broadcast_arrays(np.asarray(inc, float), np.asarray(raan, float), np.asarray(argp, float))
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)
    R = np.empty(inc.shape + (3, 3))
    R[..., 0, 0] = co * cw - so * sw * ci
    R[..., 0, 1] = -co * sw - so * cw * ci
    R[..., 0, 2] = so * si
    R[..., 1, 0] = so * cw + co * sw * ci
    R[..., 1, 1] = -so * sw + co * cw * ci
    R[..., 1, 2] = -co * si
    R[..., 2, 0] = sw * si
    R[..., 2, 1] = cw * si
    R[..., 2, 2] = ci
    return R


def par2ic(a, e, inc, raan, argp, E, mu: float = MU_MOON):
    """Orbital elements (with the eccentric anomaly E) -> inertial position and velocity (elliptic orbits)."""
    a, e, E = np.broadcast_arrays(np.asarray(a, float), np.asarray(e, float), np.asarray(E, float))
    b = a * np.sqrt(1.0 - e * e)
    n = np.sqrt(mu / a ** 3)
    cE, sE = np.cos(E), np.sin(E)
    den = 1.0 - e * cE
    x, y = a * (cE - e), b * sE                       # position in the perifocal frame
    xd, yd = -a * n * sE / den, b * n * cE / den      # velocity in the perifocal frame
    R = perifocal_rotation(inc, raan, argp)
    px, py = R[..., :, 0], R[..., :, 1]               # perifocal axes expressed in the inertial frame
    r = px * x[..., None] + py * y[..., None]
    v = px * xd[..., None] + py * yd[..., None]
    return r, v


def ic2par(r, v, mu: float = MU_MOON):
    """Inertial position and velocity -> (a, e, inc, raan, argp, E) for elliptic orbits.
    For circular or equatorial orbits the undefined angles are set to zero consistently, so
    par2ic(ic2par(r, v)) always reproduces (r, v)."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    h_hat = h / hn[..., None]
    evec = np.cross(v, h) / mu - r / rn[..., None]
    e = np.linalg.norm(evec, axis=-1)
    a = 1.0 / (2.0 / rn - np.sum(v * v, axis=-1) / mu)
    inc = np.arccos(np.clip(h_hat[..., 2], -1.0, 1.0))
    # node line (k x h); falls back to the x axis for equatorial orbits
    nvec = np.cross(np.array([0.0, 0.0, 1.0]), h)
    nn = np.linalg.norm(nvec, axis=-1)
    has_node = nn > 1e-12 * hn
    n_hat = np.where(has_node[..., None], nvec / np.where(has_node, nn, 1.0)[..., None], np.array([1.0, 0.0, 0.0]))
    raan = np.where(has_node, np.mod(np.arctan2(n_hat[..., 1], n_hat[..., 0]), TWO_PI), 0.0)
    # perifocal x axis: eccentricity direction, or the node line for circular orbits
    has_ecc = e > 1e-12
    p_hat = np.where(has_ecc[..., None], evec / np.where(has_ecc, e, 1.0)[..., None], n_hat)
    q_hat = np.cross(h_hat, p_hat)
    argp = np.where(
        has_ecc,
        np.mod(np.arctan2(np.sum(np.cross(n_hat, p_hat) * h_hat, axis=-1), np.sum(n_hat * p_hat, axis=-1)), TWO_PI),
        0.0,
    )
    nu = np.arctan2(np.sum(r * q_hat, axis=-1), np.sum(r * p_hat, axis=-1))
    E = np.mod(2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * nu), np.sqrt(1.0 + e) * np.cos(0.5 * nu)), TWO_PI)
    return a, e, inc, raan, argp, E


def propagate(r0, v0, dt_sec, mu: float = MU_MOON):
    """Move a Cartesian state along its (elliptic) Keplerian orbit by dt_sec seconds."""
    a, e, inc, raan, argp, E0 = ic2par(r0, v0, mu)
    M = E0 - e * np.sin(E0) + np.sqrt(mu / a ** 3) * np.asarray(dt_sec, float)
    E = solve_kepler(np.mod(M, TWO_PI), e)
    return par2ic(a, e, inc, raan, argp, E, mu)


def true_anomaly(E, e):
    """Eccentric anomaly -> true anomaly (elliptic orbits)."""
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E), np.sqrt(1.0 - e) * np.cos(0.5 * E))


class Ephemeris:
    """Position and velocity of every tomato at any time, vectorised over tomatoes and times.

    Times are in days from the instance epoch t0. The mean anomaly is advanced linearly,
    Kepler's equation is solved and the state is rotated into the inertial frame, which is what
    pykep's `planet.keplerian.eph` does (pykep switches to an equivalent Lagrangian propagation
    for e < 1e-3 or i < 1e-3; the results differ only at the 1e-9 relative level).
    """

    def __init__(self, instance, mu: float = MU_MOON):
        el = np.asarray(instance.elements, dtype=float)
        self.mu = mu
        self.t0 = float(instance.t0)
        self.a, self.e, self.inc, self.raan, self.argp, self.m0 = (el[:, k].copy() for k in range(6))
        self.n = np.sqrt(mu / self.a ** 3)             # mean motion [rad/s]
        self.b = self.a * np.sqrt(1.0 - self.e ** 2)   # semi-minor axis [m]
        self.R = perifocal_rotation(self.inc, self.raan, self.argp)
        self.period = TWO_PI / self.n / DAY2SEC        # [days]
        self.v_circ = np.sqrt(mu / self.a)             # [m/s]
        self.prograde = np.cos(self.inc) >= 0.0        # True when the motion is counter-clockwise seen from +z
        self.n_bodies = int(el.shape[0])

    def eccentric_anomaly(self, idx, t) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        dt = (np.asarray(t, dtype=float) - self.t0) * DAY2SEC
        M = self.m0[idx] + self.n[idx] * dt
        return solve_kepler(np.mod(M, TWO_PI), self.e[idx])

    def state(self, idx, t):
        """(r, v) of tomato(s) `idx` at time(s) `t` [days]; `idx` and `t` broadcast against each other."""
        idx = np.asarray(idx, dtype=int)
        E = self.eccentric_anomaly(idx, t)
        a, e, b, n = self.a[idx], self.e[idx], self.b[idx], self.n[idx]
        cE, sE = np.cos(E), np.sin(E)
        den = 1.0 - e * cE
        x, y = a * (cE - e), b * sE
        xd, yd = -a * n * sE / den, b * n * cE / den
        R = self.R[idx]
        px, py = R[..., :, 0], R[..., :, 1]
        r = px * x[..., None] + py * y[..., None]
        v = px * xd[..., None] + py * yd[..., None]
        return r, v

    def state_all(self, t: float):
        """(r, v) of every tomato at one time, shapes (N, 3)."""
        return self.state(np.arange(self.n_bodies), t)

    def phase(self, idx, t) -> np.ndarray:
        """Argument of latitude (argp + true anomaly) mod 2*pi: the angle of the tomato along its orbit
        measured from the ascending node. All orbits share the same node line, so phases are comparable."""
        idx = np.asarray(idx, dtype=int)
        E = self.eccentric_anomaly(idx, t)
        return np.mod(self.argp[idx] + true_anomaly(E, self.e[idx]), TWO_PI)
