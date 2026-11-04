import numpy as np
import pytest

from kttsp.constants import MU_MOON
from kttsp.kepler import Ephemeris
from kttsp.transfer import leg_dv, leg_dv_batch, transfer_angle
from tests.conftest import make_instance


def test_staying_on_the_same_orbit_costs_nothing(easy):
    eph = Ephemeris(easy)
    for idx in [0, 4, 30]:
        T = eph.period[idx]
        assert leg_dv(eph, idx, idx, 2.0, 1.37 * T) < 1e-6
        assert leg_dv(eph, idx, idx, 2.0, 0.42 * T) < 1e-6


class _FixedPositions:
    """Stand-in for Ephemeris: tomato 0 is always at r_dep and tomato 1 always at r_arr, both at rest."""
    mu = MU_MOON

    def __init__(self, r_dep, r_arr):
        self.r = [np.asarray(r_dep, dtype=float), np.asarray(r_arr, dtype=float)]

    def state(self, idx, t):
        return self.r[idx], np.zeros(3)


def test_degenerate_geometry_gives_inf(easy):
    eph = Ephemeris(easy)
    assert leg_dv(eph, 0, 1, 2.0, 0.0) == np.inf
    assert leg_dv(eph, 0, 1, 2.0, -1.0) == np.inf
    # departure and arrival on one line through the Moon's centre: no transfer plane, so no Lambert arc
    r = np.array([1.5e7, 0.0, 0.0])
    assert leg_dv(_FixedPositions(r, 2.0 * r), 0, 1, 0.0, 0.5) == np.inf
    assert leg_dv(_FixedPositions(r, -r), 0, 1, 0.0, 0.5) == np.inf


def test_pure_plane_change_cost():
    """Two circular orbits of equal radius whose planes differ by 0.1 rad: a quarter-orbit arc that
    starts on the node line costs exactly one plane change, 2*v*sin(0.05)."""
    a = 1.5e7
    v = np.sqrt(MU_MOON / a)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / MU_MOON) / 86400.0
    inst = make_instance([[a, 0.0, 0.0, 0.0, 0.0, 0.0], [a, 0.0, 0.1, 0.0, 0.0, 0.0]])
    eph = Ephemeris(inst)
    dv = leg_dv(eph, 0, 1, 0.0, 0.25 * T)
    assert dv == pytest.approx(2.0 * v * np.sin(0.05), abs=1e-6)


def test_batch_matches_scalar(easy):
    eph = Ephemeris(easy)
    rng = np.random.default_rng(2)
    K = 200
    i = rng.integers(0, easy.n, K)
    j = rng.integers(0, easy.n, K)
    t_dep = rng.uniform(0.0, 100.0, K)
    tof = rng.uniform(0.01, 30.0, K)
    dv_b = leg_dv_batch(eph, i, j, t_dep, tof)
    dv_s = np.array([leg_dv(eph, int(i[k]), int(j[k]), float(t_dep[k]), float(tof[k])) for k in range(K)])
    assert np.allclose(dv_b, dv_s, rtol=1e-12, atol=1e-9)
    assert np.sum(np.isfinite(dv_b)) > K - 5
    # restricted searches are upper bounds of the full search
    dv_one = leg_dv_batch(eph, i, j, t_dep, tof, both_cw=False)
    assert np.all(dv_one >= dv_b - 1e-9)
    dv_rev = leg_dv_batch(eph, i, j, t_dep, tof, revs=[0, 1])
    assert np.all(dv_rev >= dv_b - 1e-9)


def test_batch_broadcasting(easy):
    eph = Ephemeris(easy)
    t_dep = np.linspace(0.0, 3.0, 4)
    tof = np.linspace(0.5, 2.0, 5)
    grid = leg_dv_batch(eph, 3, 7, t_dep[:, None], tof[None, :])
    assert grid.shape == (4, 5)
    assert grid[2, 3] == pytest.approx(leg_dv(eph, 3, 7, t_dep[2], tof[3]), rel=1e-12)


def test_transfer_angle(easy):
    eph = Ephemeris(easy)
    T = eph.period[5]
    assert transfer_angle(eph, 5, 5, 1.0, 0.25 * T) == pytest.approx(0.5 * np.pi, abs=0.05)
    assert transfer_angle(eph, 5, 5, 1.0, T) == pytest.approx(0.0, abs=1e-6)
