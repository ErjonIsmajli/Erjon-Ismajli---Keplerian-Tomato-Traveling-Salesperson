import numpy as np
import pytest

from kttsp.constants import DAY2SEC, MU_MOON
from kttsp.kepler import Ephemeris, propagate
from kttsp.lambert import branch_table, lambert_batch, lambert_scalar


def _min_dv(v1s, v2s, v_dep, v_arr):
    dv = np.linalg.norm(v1s - v_dep, axis=-1) + np.linalg.norm(v_arr - v2s, axis=-1)
    return np.nanmin(dv)


def test_branch_table_order():
    revs, side = branch_table(2)
    assert revs.tolist() == [0, 1, 1, 2, 2]
    assert side.tolist() == [0, -1, 1, -1, 1]
    revs, side = branch_table(20, revs=[3, 0, 3, 25])
    assert revs.tolist() == [0, 3, 3]


def test_same_orbit_is_reproduced_for_every_revolution_count(easy):
    """r1, r2 taken on one tomato's orbit: one Lambert branch must be that orbit (zero delta-v)."""
    eph = Ephemeris(easy)
    rng = np.random.default_rng(3)
    for idx in [0, 4, 7, 23, 48]:
        T = eph.period[idx]
        for n_rev in [0, 1, 2, 5, 11]:
            frac = rng.uniform(0.15, 0.85)
            t1 = rng.uniform(0.0, 20.0)
            tof_days = (n_rev + frac) * T
            r1, v1 = eph.state(idx, t1)
            r2, v2 = eph.state(idx, t1 + tof_days)
            best = np.inf
            for cw in (False, True):
                v1s, v2s, _, revs = lambert_scalar(r1, r2, tof_days * DAY2SEC, MU_MOON, cw, 20)
                assert revs.max() >= n_rev
                best = min(best, _min_dv(v1s, v2s, v1, v2))
            assert best < 1e-6, (idx, n_rev, best)


def test_every_branch_lands_on_r2(easy):
    """Propagating (r1, v1) of any elliptic branch for the time of flight must end at (r2, v2)."""
    eph = Ephemeris(easy)
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(40):
        i, j = rng.choice(easy.n, 2, replace=False)
        t1 = rng.uniform(0.0, 30.0)
        tof_days = rng.uniform(0.05, 6.0)
        r1, _ = eph.state(i, t1)
        r2, _ = eph.state(j, t1 + tof_days)
        for cw in (False, True):
            v1s, v2s, xs, revs = lambert_scalar(r1, r2, tof_days * DAY2SEC, MU_MOON, cw, 20)
            for b in range(len(revs)):
                if not np.all(np.isfinite(v1s[b])):
                    continue
                energy = 0.5 * np.dot(v1s[b], v1s[b]) - MU_MOON / np.linalg.norm(r1)
                if energy >= 0.0:
                    continue  # hyperbolic direct arc: the Kepler propagator only handles ellipses
                r_end, v_end = propagate(r1, v1s[b], tof_days * DAY2SEC)
                assert np.allclose(r_end, r2, rtol=1e-7, atol=1.0), (i, j, cw, b)
                assert np.allclose(v_end, v2s[b], rtol=1e-7, atol=1e-4)
                checked += 1
    assert checked > 200


def test_batch_matches_scalar(easy):
    eph = Ephemeris(easy)
    rng = np.random.default_rng(11)
    K = 300
    i = rng.integers(0, easy.n, K)
    j = rng.integers(0, easy.n, K)
    t1 = rng.uniform(0.0, 50.0, K)
    tof_days = rng.uniform(0.01, 40.0, K)
    r1, _ = eph.state(i, t1)
    r2, _ = eph.state(j, t1 + tof_days)
    cw = rng.integers(0, 2, K).astype(bool)
    v1b, v2b, revs = lambert_batch(r1, r2, tof_days * DAY2SEC, MU_MOON, cw, 20)
    assert v1b.shape == (K, 41, 3) and revs.shape == (41,)
    n_compared = 0
    for k in range(K):
        try:
            v1s, v2s, _, revs_s = lambert_scalar(r1[k], r2[k], tof_days[k] * DAY2SEC, MU_MOON, bool(cw[k]), 20)
        except ValueError:
            assert np.all(np.isnan(v1b[k]))
            continue
        nb = len(revs_s)
        assert np.allclose(v1b[k, :nb], v1s, rtol=1e-12, atol=1e-9)
        assert np.allclose(v2b[k, :nb], v2s, rtol=1e-12, atol=1e-9)
        assert np.all(np.isnan(v1b[k, nb:]))
        n_compared += nb
    assert n_compared > 2000


def test_restricted_revolutions_subset(easy):
    eph = Ephemeris(easy)
    r1, _ = eph.state(np.array([1, 2]), np.array([0.0, 1.0]))
    r2, _ = eph.state(np.array([3, 4]), np.array([5.0, 6.0]))
    tof = np.array([5.0, 5.0]) * DAY2SEC
    full_v1, full_v2, full_revs = lambert_batch(r1, r2, tof, MU_MOON, False, 20)
    part_v1, part_v2, part_revs = lambert_batch(r1, r2, tof, MU_MOON, False, 20, revs=[1, 2])
    assert part_revs.tolist() == [1, 1, 2, 2]
    sel = [1, 2, 3, 4]
    assert np.allclose(part_v1, full_v1[:, sel], equal_nan=True)
    assert np.allclose(part_v2, full_v2[:, sel], equal_nan=True)


def test_parallel_positions_are_rejected():
    r1 = np.array([1.5e7, 0.0, 0.0])
    with pytest.raises(ValueError):
        lambert_scalar(r1, 2.0 * r1, 1000.0, MU_MOON)
    with pytest.raises(ValueError):
        lambert_scalar(r1, r1, -1.0, MU_MOON)
    v1, v2, _ = lambert_batch(np.vstack([r1, r1]), np.vstack([2.0 * r1, -r1]), np.array([1000.0, 1000.0]), MU_MOON)
    assert np.all(np.isnan(v1)) and np.all(np.isnan(v2))


def test_circular_quarter_orbit():
    """A quarter revolution on a circular orbit is the direct Lambert arc with circular speed."""
    a = 1.5e7
    v_c = np.sqrt(MU_MOON / a)
    quarter = 0.25 * 2.0 * np.pi * np.sqrt(a ** 3 / MU_MOON)
    r1 = np.array([a, 0.0, 0.0])
    r2 = np.array([0.0, a, 0.0])
    v1, v2, xs, revs = lambert_scalar(r1, r2, quarter, MU_MOON, False, 20)
    assert revs.tolist() == [0]
    assert np.allclose(v1[0], [0.0, v_c, 0.0], atol=1e-6)
    assert np.allclose(v2[0], [-v_c, 0.0, 0.0], atol=1e-6)
    # the retrograde solution goes the long way round and is much faster
    v1r, _, _, _ = lambert_scalar(r1, r2, quarter, MU_MOON, True, 20)
    assert v1r[0, 1] < 0.0 and np.linalg.norm(v1r[0]) > v_c
