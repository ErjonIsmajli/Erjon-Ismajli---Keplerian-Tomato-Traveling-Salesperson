"""Loading of .kttsp instance files and geometric helpers on the tomato orbits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import DAY2SEC, MU_MOON


class InstanceError(ValueError):
    """An instance file that cannot be read or describes an impossible problem."""


@dataclass
class Instance:
    """One KTTSP problem: the mission parameters plus the orbital elements of every tomato."""

    name: str
    t0: float             # mission start time [days]
    min_tof: float        # minimum time of flight of one transfer [days]
    t_max: float          # maximum total mission time [days]
    dv_max: float         # delta-v limit of a normal transfer [m/s]
    dv_exception: float   # delta-v limit of an exception transfer [m/s]
    n_exceptions: int     # number of exception transfers allowed
    elements: np.ndarray  # (N, 6): a [m], e, i [rad], RAAN [rad], argp [rad], mean anomaly at t0 [rad]

    # ---- per-tomato quantities -------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.elements.shape[0])

    @property
    def a(self) -> np.ndarray:
        return self.elements[:, 0]

    @property
    def e(self) -> np.ndarray:
        return self.elements[:, 1]

    @property
    def inc(self) -> np.ndarray:
        return self.elements[:, 2]

    @property
    def raan(self) -> np.ndarray:
        return self.elements[:, 3]

    @property
    def argp(self) -> np.ndarray:
        return self.elements[:, 4]

    @property
    def m0(self) -> np.ndarray:
        return self.elements[:, 5]

    def period(self) -> np.ndarray:
        """Orbital period of every tomato [days]."""
        return 2.0 * np.pi * np.sqrt(self.a ** 3 / MU_MOON) / DAY2SEC

    def v_circ(self) -> np.ndarray:
        """Circular velocity sqrt(mu/a) of every tomato [m/s]; the orbits are nearly circular."""
        return np.sqrt(MU_MOON / self.a)

    def orbit_normals(self) -> np.ndarray:
        """Unit vector along the angular momentum of every orbit, (N, 3)."""
        si, ci = np.sin(self.inc), np.cos(self.inc)
        return np.stack([np.sin(self.raan) * si, -np.cos(self.raan) * si, ci], axis=1)

    def h_z_sign(self) -> np.ndarray:
        """+1 for orbits that are prograde as seen from +z, -1 for retrograde ones."""
        return np.where(np.cos(self.inc) >= 0.0, 1, -1)

    def relative_inclination_matrix(self) -> np.ndarray:
        """Angle between the orbital planes of every pair of tomatoes [rad], (N, N)."""
        h = self.orbit_normals()
        return np.arccos(np.clip(h @ h.T, -1.0, 1.0))

    def plane_change_matrix(self) -> np.ndarray:
        """Delta-v of a pure plane change from the orbit of tomato i into the plane of tomato j,
        done at the circular speed of i [m/s], (N, N)."""
        return 2.0 * self.v_circ()[:, None] * np.sin(0.5 * self.relative_inclination_matrix())

    @staticmethod
    def max_plane_change(dv: float, v: float) -> float:
        """Largest plane-change angle [rad] that a delta-v budget buys at speed v."""
        return float(2.0 * np.arcsin(min(1.0, dv / (2.0 * v))))

    # ---- structure of the instance ---------------------------------------------------

    def groups(self, split_radius: float = 8.0e6) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the low-altitude and of the high-altitude tomatoes (split at `split_radius` metres)."""
        low = np.flatnonzero(self.a < split_radius)
        high = np.flatnonzero(self.a >= split_radius)
        return low, high

    def clusters(self, indices, max_gap: float) -> list[np.ndarray]:
        """Split `indices` into clusters of neighbouring inclinations: sort by inclination and cut
        wherever two consecutive tomatoes differ by more than `max_gap` radians."""
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return []
        order = idx[np.argsort(self.inc[idx])]
        cuts = np.flatnonzero(np.diff(self.inc[order]) > max_gap) + 1
        return [np.asarray(c) for c in np.split(order, cuts)]


def _number(text: str, line: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise InstanceError(f"not a number: {text!r} in line {line!r}") from None
    if not np.isfinite(value):
        raise InstanceError(f"not a finite number: {text!r} in line {line!r}")
    return value


def parse_instance(text: str, name: str = "instance") -> Instance:
    """Parse the text of a .kttsp file: comment lines start with 'c', the parameter line with 'p',
    every other non-empty line holds the six orbital elements of one tomato. Raises InstanceError
    for malformed files and for values that cannot describe a valid problem."""
    header = None
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) != 8 or parts[1].lower() != "kttsp":
                raise InstanceError(f"malformed parameter line: {line!r}")
            header = [_number(v, line) for v in parts[2:]]
            continue
        parts = line.split()
        if len(parts) != 6:
            raise InstanceError(f"expected 6 orbital elements per line, got: {line!r}")
        row = [_number(v, line) for v in parts]
        if row[0] <= 0.0:
            raise InstanceError(f"semi-major axis must be positive: {line!r}")
        if not 0.0 <= row[1] < 1.0:
            raise InstanceError(f"eccentricity must be in [0, 1): {line!r}")
        rows.append(row)
    if header is None:
        raise InstanceError("missing 'p kttsp ...' parameter line")
    if len(rows) < 2:
        raise InstanceError(f"an instance needs at least two tomatoes, found {len(rows)}")
    t0, min_tof, t_max, dv_max, dv_exception, n_exc = header
    if min_tof < 0.0 or t_max <= t0:
        raise InstanceError("the minimum time of flight must be >= 0 and t_max must be after t0")
    if dv_max <= 0.0 or dv_exception < dv_max:
        raise InstanceError("the delta-v limit must be positive and the exception limit at least as large")
    if n_exc < 0 or n_exc != int(n_exc):
        raise InstanceError(f"the number of exceptions must be a non-negative integer, got {n_exc}")
    return Instance(name=name, t0=t0, min_tof=min_tof, t_max=t_max, dv_max=dv_max, dv_exception=dv_exception,
                    n_exceptions=int(n_exc), elements=np.asarray(rows, dtype=float))


def load_instance(path) -> Instance:
    """Load a .kttsp file from disk; the instance name is the file stem (easy, medium, hard)."""
    path = Path(path)
    return parse_instance(path.read_text(encoding="utf-8"), name=path.stem)
