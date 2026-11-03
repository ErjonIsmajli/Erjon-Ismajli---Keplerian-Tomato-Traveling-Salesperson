import numpy as np
import pytest

from kttsp.constants import DAY2SEC, MU_MOON, TWO_PI
from kttsp.kepler import Ephemeris, ic2par, par2ic, propagate, solve_kepler
from tests.conftest import make_instance


def test_kepler_equation_residual():
    rng = np.random.default_rng(1)
    M = rng.uniform(0.0, TWO_PI, 500)
    e = rng.uniform(0.0, 0.95, 500)
    E = solve_kepler(M, e)
    assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-12


def test_kepler_equation_converges_for_very_eccentric_orbits():
    """Plain Newton steps diverge for some of these; the solver must converge for every e < 1."""
    rng = np.random.default_rng(2)
    M = rng.uniform(0.0, TWO_PI, 50000)
    e = rng.uniform(0.5, 0.9999, 50000)
    E = solve_kepler(M, e)
    assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-9
    assert float(solve_kepler(1.0, 0.99)) == pytest.approx(1.92763555, abs=1e-8)


def test_circular_orbit_radius_and_speed():
    a = 1.5e7
    inst = make_instance([[a, 0.0, 0.3, 0.1, 0.2, 0.5]])
    eph = Ephemeris(inst)
    t = np.linspace(0.0, 10.0, 50)
    r, v = eph.state(0, t)
    assert r.shape == (50, 3) and v.shape == (50, 3)
    assert np.allclose(np.linalg.norm(r, axis=1), a, rtol=1e-12)
    assert np.allclose(np.linalg.norm(v, axis=1), np.sqrt(MU_MOON / a), rtol=1e-12)
    assert np.allclose(np.sum(r * v, axis=1), 0.0, atol=1e-3)


def test_state_after_one_period_returns_to_start(easy):
    eph = Ephemeris(easy)
    for idx in [0, 4, 10, 25, 48]:
        r0, v0 = eph.state(idx, 3.0)
        r1, v1 = eph.state(idx, 3.0 + eph.period[idx])
        assert np.allclose(r0, r1, rtol=1e-9, atol=1e-3)
        assert np.allclose(v0, v1, rtol=1e-9, atol=1e-7)


def test_state_broadcasting(easy):
    eph = Ephemeris(easy)
    r, v = eph.state(3, 1.5)
    assert r.shape == (3,) and v.shape == (3,)
    idx = np.array([1, 2, 3])
    r, v = eph.state(idx, np.array([0.5, 1.5, 2.5]))
    assert r.shape == (3, 3)
    for k in range(3):
        rk, _ = eph.state(idx[k], [0.5, 1.5, 2.5][k])
        assert np.allclose(rk, r[k])
    r_all, v_all = eph.state_all(2.0)
    assert r_all.shape == (easy.n, 3)
    assert np.allclose(r_all[3], eph.state(3, 2.0)[0])


def test_energy_and_angular_momentum_conserved(easy):
    eph = Ephemeris(easy)
    t = np.array([0.0, 0.7, 13.1, 199.0, 2999.0])
    for idx in [2, 4, 11, 30]:
        r, v = eph.state(idx, t)
        energy = 0.5 * np.sum(v * v, axis=1) - MU_MOON / np.linalg.norm(r, axis=1)
        h = np.cross(r, v)
        assert np.allclose(energy, energy[0], rtol=1e-10)
        assert np.allclose(h, h[0], rtol=1e-10)


def test_par2ic_ic2par_roundtrip_of_elements():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a = rng.uniform(2e6, 2e7)
        e = rng.uniform(0.01, 0.8)
        inc = rng.uniform(0.05, np.pi - 0.05)
        raan = rng.uniform(0.0, TWO_PI)
        argp = rng.uniform(0.0, TWO_PI)
        E = rng.uniform(0.0, TWO_PI)
        r, v = par2ic(a, e, inc, raan, argp, E)
        a2, e2, inc2, raan2, argp2, E2 = ic2par(r, v)
        assert a2 == pytest.approx(a, rel=1e-10)
        assert e2 == pytest.approx(e, abs=1e-10)
        assert inc2 == pytest.approx(inc, abs=1e-10)
        assert np.mod(raan2 - raan + np.pi, TWO_PI) - np.pi == pytest.approx(0.0, abs=1e-9)
        assert np.mod(argp2 - argp + np.pi, TWO_PI) - np.pi == pytest.approx(0.0, abs=1e-8)
        assert np.mod(E2 - E + np.pi, TWO_PI) - np.pi == pytest.approx(0.0, abs=1e-8)


def test_ic2par_par2ic_roundtrip_of_states(easy):
    """Includes tomatoes with tiny eccentricity and negative inclination."""
    eph = Ephemeris(easy)
    r, v = eph.state_all(5.3)
    a, e, inc, raan, argp, E = ic2par(r, v)
    r2, v2 = par2ic(a, e, inc, raan, argp, E)
    # 1 cm / 1e-6 m/s tolerance: orbits with i = pi have an undefined node line (tiny frame noise)
    assert np.allclose(r, r2, rtol=1e-9, atol=1e-2)
    assert np.allclose(v, v2, rtol=1e-9, atol=1e-6)
    assert np.allclose(a, easy.a, rtol=1e-9)
    assert np.allclose(e, easy.e, atol=1e-9)


def test_negative_inclination_equals_rotated_positive_one():
    """A rotation by -i about the node line is the same orbit as +i with RAAN and argp shifted by pi."""
    a, e, i, raan, argp, m0 = 1.5e7, 0.005, -0.02, 0.004, 1.2, 2.5
    eph_neg = Ephemeris(make_instance([[a, e, i, raan, argp, m0]]))
    eph_pos = Ephemeris(make_instance([[a, e, -i, raan + np.pi, argp + np.pi, m0]]))
    t = np.linspace(0.0, 5.0, 20)
    r1, v1 = eph_neg.state(0, t)
    r2, v2 = eph_pos.state(0, t)
    assert np.allclose(r1, r2, rtol=1e-10, atol=1e-3)
    assert np.allclose(v1, v2, rtol=1e-10, atol=1e-7)


def test_propagate_matches_ephemeris(easy):
    eph = Ephemeris(easy)
    for idx, dt_days in [(0, 0.3), (4, 2.7), (20, 15.0), (48, 0.05)]:
        r0, v0 = eph.state(idx, 1.0)
        r1, v1 = propagate(r0, v0, dt_days * DAY2SEC)
        r_ref, v_ref = eph.state(idx, 1.0 + dt_days)
        assert np.allclose(r1, r_ref, rtol=1e-9, atol=1e-3)
        assert np.allclose(v1, v_ref, rtol=1e-9, atol=1e-7)


def test_phase_advances_at_the_mean_motion_for_circular_orbits():
    a = 1.5e7
    inst = make_instance([[a, 0.0, 0.3, 0.0, 0.4, 1.0]])
    eph = Ephemeris(inst)
    dt = 0.1  # days
    p0 = eph.phase(0, 2.0)
    p1 = eph.phase(0, 2.0 + dt)
    expected = eph.n[0] * dt * DAY2SEC
    assert np.mod(p1 - p0, TWO_PI) == pytest.approx(expected, abs=1e-9)
    # the phase is measured from the ascending node: argp + true anomaly
    assert p0 == pytest.approx(np.mod(0.4 + 1.0 + eph.n[0] * 2.0 * DAY2SEC, TWO_PI), abs=1e-9)
