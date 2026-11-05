import json

import numpy as np
import pytest

from kttsp.constants import MU_MOON
from kttsp.evaluator import evaluate, split_chromosome
from kttsp.kepler import Ephemeris
from kttsp.solution import CHALLENGE_ID, Solution
from kttsp.transfer import leg_dv
from tests.conftest import make_instance

A = 1.5e7
T = 2.0 * np.pi * np.sqrt(A ** 3 / MU_MOON) / 86400.0   # period [days]
BASE = [A, 0.002, 0.5, 0.01, 1.0, 0.3]


def three_identical_orbits(**params):
    """Three tomatoes sharing one orbit and phase: moving between them costs no delta-v."""
    return make_instance([BASE, BASE, BASE], **params)


def feasible_chromosome():
    # leg 0: 0 -> 1 departing at 0.0, leg 1: 1 -> 2 departing 0.5 d after the first arrival
    t_dep = [0.0, 1.3 * T + 0.5]
    tof = [1.3 * T, 1.3 * T]
    return t_dep + tof + [0, 1, 2]


def test_feasible_tour_is_accepted():
    inst = three_identical_orbits()
    rep = evaluate(inst, feasible_chromosome())
    assert rep.feasible, rep.violations
    assert rep.objective == pytest.approx(2.6 * T + 0.5)
    assert rep.n_exceptions == 0 and len(rep.legs) == 2
    assert all(leg.dv < 1e-6 for leg in rep.legs)
    assert "feasible: True" in rep.summary()


def test_wrong_length_and_non_permutation():
    inst = three_identical_orbits()
    rep = evaluate(inst, feasible_chromosome()[:-1])
    assert not rep.feasible and "expected 3N-2" in rep.violations[0] and rep.objective == np.inf
    x = feasible_chromosome()
    x[-1] = 1
    rep = evaluate(inst, x)
    assert not rep.feasible and "permutation" in rep.violations[0]
    x = feasible_chromosome()
    x[-1] = 2.5
    rep = evaluate(inst, x)
    assert not rep.feasible and any("non-integer" in v for v in rep.violations)


def test_time_ordering_and_minimum_tof():
    inst = three_identical_orbits()
    x = feasible_chromosome()
    x[1] = 1.0  # second departure before the first arrival
    rep = evaluate(inst, x)
    assert not rep.feasible and any("before the arrival" in v for v in rep.violations)
    x = feasible_chromosome()
    x[2] = 0.0005  # below min_tof = 0.001
    rep = evaluate(inst, x)
    assert not rep.feasible and any("below the minimum" in v for v in rep.violations)
    x = feasible_chromosome()
    x[0] = -0.1
    rep = evaluate(inst, x)
    assert not rep.feasible and any("before the start time" in v for v in rep.violations)


def test_mission_time_limit():
    inst = three_identical_orbits(t_max=3.0)
    rep = evaluate(inst, feasible_chromosome())
    assert not rep.feasible and any("maximum time" in v for v in rep.violations)
    assert rep.objective == pytest.approx(2.6 * T + 0.5)   # the objective is still reported


def test_exception_counting_and_limits():
    # tomato 2 sits in a plane 0.4 rad away (a plane change of about 227 m/s: an exception),
    # tomato 3 in a plane 1.5 rad away (the leg 2 -> 3 below, flown in one period, costs about 1250 m/s)
    far = list(BASE)
    far[2] += 0.4
    too_far = list(BASE)
    too_far[2] += 1.5
    inst = make_instance([BASE, BASE, far, too_far], n_exceptions=1)
    eph = Ephemeris(inst)
    # pick a time of flight for leg 1 -> 2 whose cost is within the exception limit
    tofs = np.linspace(0.3 * T, 3.0 * T, 300)
    costs = np.array([leg_dv(eph, 1, 2, 1.3 * T, tof) for tof in tofs])
    k = int(np.argmin(costs))
    assert inst.dv_max < costs[k] <= inst.dv_exception
    t_dep = [0.0, 1.3 * T, 1.3 * T + tofs[k]]
    tof = [1.3 * T, tofs[k], 1.0 * T]
    x = t_dep + tof + [0, 1, 2, 3]
    rep = evaluate(inst, x)
    assert rep.n_exceptions >= 1
    assert any("exceeds the exception limit" in v for v in rep.violations)
    assert rep.legs[1].is_exception and rep.legs[2].dv > inst.dv_exception
    # with the last tomato removed the tour uses one exception, which is allowed ...
    inst3 = make_instance([BASE, BASE, far], n_exceptions=1)
    rep = evaluate(inst3, t_dep[:2] + tof[:2] + [0, 1, 2])
    assert rep.feasible and rep.n_exceptions == 1
    # ... but not when no exception is allowed
    inst3.n_exceptions = 0
    rep = evaluate(inst3, t_dep[:2] + tof[:2] + [0, 1, 2])
    assert not rep.feasible and any("exception transfers used" in v for v in rep.violations)


def test_minimum_time_of_flight_override():
    inst = three_identical_orbits()
    x = feasible_chromosome()
    x[2] = 0.005   # first time of flight: allowed by the file (0.001 d), not by a 0.01 d minimum
    assert evaluate(inst, x).feasible
    rep = evaluate(inst, x, min_tof=0.01)
    assert not rep.feasible and any("below the minimum 0.010000" in v for v in rep.violations)


def test_malformed_vectors_are_reported_not_raised():
    inst = three_identical_orbits()
    for bad in ("abc", None, [[1.0, 2.0], [3.0, 4.0]]):
        rep = evaluate(inst, bad)
        assert not rep.feasible and rep.objective == np.inf
    x = feasible_chromosome()
    x[-1] = float("nan")
    rep = evaluate(inst, x)
    assert not rep.feasible and "not finite" in rep.violations[0]


def test_single_tomato_instance_is_reported():
    inst = make_instance([BASE])
    rep = evaluate(inst, [0])
    assert not rep.feasible and "at least two" in rep.violations[0]


def test_times_must_be_finite():
    inst = three_identical_orbits()
    x = feasible_chromosome()
    x[1] = float("nan")
    rep = evaluate(inst, x)
    assert not rep.feasible and "finite" in rep.violations[0]


def test_injected_transfer_model():
    inst = three_identical_orbits()
    rep = evaluate(inst, feasible_chromosome(), dv_fn=lambda i, j, t, dt: 42.0)
    assert rep.feasible and all(leg.dv == 42.0 for leg in rep.legs)


def test_split_chromosome():
    t, tof, order = split_chromosome(feasible_chromosome(), 3)
    assert t.tolist() == [0.0, 1.3 * T + 0.5] and order.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        split_chromosome([1, 2, 3], 3)


def test_solution_roundtrip_and_files(tmp_path):
    sol = Solution([2, 0, 1], [0.0, 3.0], [1.0, 1.5], instance="easy", dv=[10.0, 20.0], objective=4.5,
                   exceptions=[1], meta={"solver": "test"})
    x = sol.to_chromosome()
    assert x == [0.0, 3.0, 1.0, 1.5, 2, 0, 1]
    back = Solution.from_chromosome(x, 3, instance="easy")
    assert back.order.tolist() == [2, 0, 1] and back.t_end == 4.5
    sol.save(tmp_path / "s.json")
    loaded = Solution.load(tmp_path / "s.json")
    assert loaded.order.tolist() == [2, 0, 1] and loaded.exceptions == [1] and loaded.meta == {"solver": "test"}
    assert loaded.objective == 4.5 and loaded.dv.tolist() == [10.0, 20.0]
    sol.write_submission(tmp_path / "sub.json", name="run 1", description="test")
    sub = json.loads((tmp_path / "sub.json").read_text())
    assert sub[0]["challenge"] == CHALLENGE_ID and sub[0]["problem"] == "small"
    assert sub[0]["decisionVector"] == x and sub[0]["name"] == "run 1"
