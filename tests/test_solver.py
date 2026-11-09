import copy

import numpy as np
import pytest

from kttsp.construct import Block, ConstructSettings, GreedyBuilder, analyse, construct, make_plans
from kttsp.evaluator import evaluate
from kttsp.improve import Annealer, Decoder, Tour, improve, retime_polish
from kttsp.solution import Solution
from kttsp.kepler import Ephemeris
from kttsp.leg_search import LegSearcher, SearchSettings
from tests.conftest import make_instance

A_HIGH, A_LOW = 1.5e7, 3.75e6


def small_instance():
    """Four high tomatoes along an inclination chain plus a low cluster of two near i = 0."""
    rows = []
    rng = np.random.default_rng(0)
    for k, inc in enumerate([0.0, 0.08, 0.16, 0.24]):
        rows.append([A_HIGH * (1 + 0.002 * k), 0.003, inc, 0.002 * k, rng.uniform(0, 6.28), rng.uniform(0, 6.28)])
    for k, inc in enumerate([0.01, 0.05]):
        rows.append([A_LOW * (1 + 0.001 * k), 0.002, inc, 0.001, rng.uniform(0, 6.28), rng.uniform(0, 6.28)])
    return make_instance(rows, t_max=60.0, n_exceptions=2)


def test_structure_and_plans():
    inst = small_instance()
    st = analyse(inst)
    assert st.high_chain.tolist() == [0, 1, 2, 3]
    assert len(st.low_clusters) == 1 and sorted(st.low_clusters[0].tolist()) == [4, 5]
    plans = make_plans(inst, st, ConstructSettings())
    assert plans, "at least one plan"
    for plan in plans:
        covered = np.concatenate([b.indices for b in plan])
        assert sorted(covered.tolist()) == list(range(inst.n))
        assert sum(b.exception_in for b in plan) <= inst.n_exceptions


@pytest.fixture(scope="module")
def small_solution():
    """The synthetic instance and one constructed tour, built once for all tests of this file."""
    inst = small_instance()
    eph = Ephemeris(inst)
    searcher = LegSearcher(eph, inst, SearchSettings())
    sol = construct(inst, eph, searcher, ConstructSettings(max_starts=1, both_directions=False), log=lambda *a: None)
    return inst, eph, searcher, sol


def test_construct_and_improve_small_instance(small_solution):
    inst, eph, searcher, sol = small_solution
    sol = copy.deepcopy(sol)
    assert sol is not None
    report = evaluate(inst, sol.to_chromosome(), eph)
    assert report.feasible, report.violations
    assert report.n_exceptions <= inst.n_exceptions
    assert sorted(sol.order.tolist()) == list(range(inst.n))
    sol.meta["checkpoint"] = True
    messages = []
    better = improve(inst, sol, eph, searcher, time_budget=12.0, log=messages.append)
    # the annealing runs only when more than 10 s of the budget are left: check that it did run
    assert any(m.strip().startswith("SA done") and "tried 0," not in m for m in messages), messages
    report2 = evaluate(inst, better.to_chromosome(), eph)
    assert report2.feasible, report2.violations
    assert report2.objective <= report.objective + 1e-9
    assert "checkpoint" not in better.meta


def test_figures_are_written(small_solution, tmp_path):
    from kttsp.plotting import save_all_figures
    from kttsp.solution import Solution

    inst, _, _, sol = small_solution
    paths = save_all_figures(inst, sol, tmp_path)
    assert len(paths) == 5 and all(p.exists() and p.stat().st_size > 1000 for p in paths)
    sol.save(tmp_path / "tour.json")
    loaded = Solution.load(tmp_path / "tour.json")
    assert loaded.n == inst.n and loaded.order.tolist() == sol.order.tolist()
    # a tour known only by its decision vector (no stored delta-v) can be plotted too
    bare = Solution.from_chromosome(sol.to_chromosome(), inst.n, instance=inst.name)
    paths = save_all_figures(inst, bare, tmp_path / "bare")
    assert len(paths) == 5 and all(p.exists() for p in paths)


def test_polish_searches_again_and_recovers_a_delayed_departure(small_solution):
    inst, eph, searcher, sol = small_solution
    tour = Tour.from_solution(sol)
    first = tour.legs[0]
    limit = inst.dv_exception if first.dv > inst.dv_max else inst.dv_max
    late = searcher.earliest_arrival(first.i, first.j, inst.t0 + 3.0, limit)
    assert late is not None and late.t_arr > first.t_arr
    rest = Decoder(inst, eph, searcher).decode(tour.order, None, start=1, prefix=[late])
    delayed = Tour(tour.order, [late] + rest)
    polished = retime_polish(delayed, inst, eph, base_settings=searcher.s, log=lambda *a: None)
    assert polished.legs[0].t_arr < delayed.legs[0].t_arr - 1e-6
    assert polished.objective <= delayed.objective + 1e-9
    assert evaluate(inst, polished.to_solution(inst).to_chromosome(), eph).feasible


def test_stricter_minimum_time_of_flight_is_respected(easy):
    eph = Ephemeris(easy)
    cluster = analyse(easy).low_clusters[0]
    i, j = int(cluster[0]), int(cluster[1])
    loose = LegSearcher(eph, easy, SearchSettings()).earliest_arrival(i, j, 5.0, easy.dv_max)
    strict_searcher = LegSearcher(eph, easy, SearchSettings(min_tof_floor=0.3))
    strict = strict_searcher.earliest_arrival(i, j, 5.0, easy.dv_max)
    assert loose is not None and strict is not None
    assert strict.tof >= 0.3 and strict.t_arr >= loose.t_arr - 1e-9
    assert strict_searcher.tof_min > 0.3


def test_decoder_does_not_reuse_legs_shorter_than_the_minimum(small_solution):
    inst, eph, _, sol = small_solution
    tour = Tour.from_solution(sol)
    floor = min(leg.tof for leg in tour.legs) + 0.01
    strict = Decoder(inst, eph, LegSearcher(eph, inst, SearchSettings(min_tof_floor=floor)))
    legs = strict.decode(tour.order, tour.legs, start=0)
    assert legs is not None
    assert all(leg.tof >= floor for leg in legs)


def test_cache_is_reused_only_for_a_later_start(easy):
    eph = Ephemeris(easy)
    chain = analyse(easy).high_chain
    i, j = int(chain[10]), int(chain[11])
    T = eph.period[i]
    searcher = LegSearcher(eph, easy)
    t_early, t_late = 20.0 * T + 0.01 * T, 20.0 * T + 0.09 * T     # same tenth of a period
    searcher.earliest_arrival(i, j, t_late, easy.dv_max)
    n = searcher.n_searches
    searcher.earliest_arrival(i, j, t_early, easy.dv_max)
    assert searcher.n_searches == n + 1          # an earlier start needs a new search
    searcher.earliest_arrival(i, j, t_early, easy.dv_max)
    assert searcher.n_searches == n + 1          # the same start is served from the cache


def test_spare_exceptions_are_kept_for_later_cluster_entries():
    plan = [Block("low-0", np.array([4, 5]), 0, False), Block("high-1", np.array([0, 1]), 1, True),
            Block("low-1", np.array([2, 3]), 0, True)]
    assert GreedyBuilder._spare_exceptions(plan, 0, 2) == 0
    assert GreedyBuilder._spare_exceptions(plan, 0, 3) == 1
    assert GreedyBuilder._spare_exceptions(plan, 1, 2) == 1
    assert GreedyBuilder._spare_exceptions(plan, 2, 1) == 1


def test_parallel_construction_records_search_statistics():
    inst = small_instance()
    sol = construct(inst, settings=ConstructSettings(max_starts=1, workers=2), log=lambda *a: None)
    assert sol is not None
    assert sol.meta["leg_searches"] > 0 and sol.meta["grid_points"] > 0


def test_exceptions_are_reserved_for_altitude_changes(small_solution):
    inst, eph, searcher, sol = small_solution
    dec = Decoder(inst, eph, searcher)
    order = np.array([4, 0, 1, 5, 2, 3])            # low, high, high, low, high, high
    assert dec.required_after(order).tolist() == [2, 2, 1, 0, 0]
    # a leg inside the high group may not use an exception that a later altitude change needs
    leg = dec.leg(0, 3, 0.0, exc_left=1, exc_reserved=1)
    assert leg is None or leg.dv <= inst.dv_max


def test_moves_keep_a_permutation_on_tiny_tours(small_solution):
    inst, eph, searcher, _ = small_solution
    ann = Annealer(inst, Decoder(inst, eph, searcher), np.random.default_rng(1), window=5)
    for n in (2, 3, 4, 6):
        order = np.arange(n)
        for _ in range(300):
            res = ann.propose(order)
            if res is not None:
                new, lo, hi = res
                assert sorted(new.tolist()) == list(range(n)) and 0 <= lo <= hi < n


def test_improve_accepts_a_file_without_transfer_costs(small_solution):
    inst, eph, searcher, sol = small_solution
    bare = Solution.from_chromosome(sol.to_chromosome(), inst.n, instance=inst.name)
    assert bare.dv is None
    out = improve(inst, bare, eph, searcher, time_budget=0.0, log=lambda *a: None)
    assert evaluate(inst, out.to_chromosome(), eph).feasible
    assert out.t_end <= sol.t_end + 1e-9
