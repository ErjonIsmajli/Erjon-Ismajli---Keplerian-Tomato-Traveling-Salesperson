import numpy as np
import pytest

from kttsp.instance import load_instance, parse_instance
from tests.conftest import DATA

# The example instance printed in the official challenge description.
SAMPLE = """c starting time [d], minimum time of flight [d], maximum total time [d], maximum delta-v [m/s], exception threshold [m/s] and number of exceptions
p kttsp 0.0 0.001 150.0 100.0 600.0 5
c
c The orbital parameters of the tomatoes:
1.496861100e+07 7.282163486e-03 1.053864218e+00 3.677831327e-03 3.972894704e+00 3.980584570e+00
1.497827726e+07 1.158690595e-03 1.593610788e+00 6.232981268e-03 2.079093608e+00 3.993488927e-01
1.497495376e+07 1.694927467e-03 3.037206232e+00 5.568012625e-03 5.882033922e+00 4.373284192e+00
1.498069418e+07 2.713490318e-03 2.065518687e-02 3.567533267e-03 1.765163584e+00 3.409860056e+00
3.746038900e+06 2.912291402e-03 1.577824251e+00 1.394938607e-03 1.835598963e+00 2.301919351e+00
1.500157063e+07 8.492234105e-03 2.306501279e+00 6.576128923e-03 3.570788266e+00 5.885759249e-01
1.497660921e+07 9.429097039e-03 5.319321089e-01 3.232029320e-03 3.259657612e+00 4.417198393e+00
1.501424963e+07 6.334037565e-03 1.143864218e-01 8.714605902e-03 5.049620585e+00 1.172254253e+00
1.495371830e+07 2.786464642e-03 8.450913743e-01 9.082658860e-03 1.505211752e+00 9.104013314e-01
1.500024812e+07 3.390297910e-03 1.784569170e+00 3.492095746e-03 4.561314055e+00 5.636710004e+00
1.499520489e+07 9.717649377e-04 3.141592654e+00 6.150072267e-03 6.220691804e+00 8.801738263e-01
3.749550897e+06 1.705241237e-03 1.543467712e+00 9.488855373e-03 6.067245002e+00 5.079310340e+00
"""


@pytest.mark.parametrize(
    "name,n,n_low,min_tof,t_max",
    [("easy", 49, 9, 0.001, 200.0), ("medium", 181, 60, 0.001, 500.0), ("hard", 1051, 450, 0.001, 3000.0)],
)
def test_official_files(name, n, n_low, min_tof, t_max):
    inst = load_instance(DATA / f"{name}.kttsp")
    assert inst.name == name
    assert inst.n == n
    assert (inst.t0, inst.min_tof, inst.t_max) == (0.0, min_tof, t_max)
    assert (inst.dv_max, inst.dv_exception, inst.n_exceptions) == (100.0, 600.0, 5)
    low, high = inst.groups()
    assert len(low) == n_low and len(low) + len(high) == n
    assert np.all(inst.e < 0.011) and np.all(inst.e >= 0.0)


def test_sample_from_challenge_page():
    inst = parse_instance(SAMPLE, name="sample")
    assert inst.n == 12
    assert inst.t_max == 150.0 and inst.min_tof == 0.001
    assert inst.elements[0, 0] == 1.496861100e07
    assert inst.elements[-1, 5] == 5.079310340e00


def test_crlf_line_endings_are_accepted():
    inst = parse_instance(SAMPLE.replace("\n", "\r\n"))
    assert inst.n == 12


@pytest.mark.parametrize(
    "text",
    [
        "p kttsp 0.0 0.001 150.0 100.0\n1 2 3 4 5 6\n",           # old 6-field header
        "p other 0 0.001 150 100 600 5\n1 2 3 4 5 6\n",           # wrong problem tag
        "c only comments\n",                                       # no header, no rows
        "p kttsp 0 0.001 150 100 600 5\n1 2 3 4 5\n",              # 5 elements in a row
        "p kttsp 0 0.001 150 100 600 5\n1.5e7 0 0 0 0 0\n",        # a single tomato
        "p kttsp 0 0.001 150 100 600 5\n1.5e7 0 0 0 0 0\n1.5e7 nan 0 0 0 0\n",   # not a finite number
        "p kttsp 0 0.001 150 100 600 5\n1.5e7 0 0 0 0 0\n-1.5e7 0 0 0 0 0\n",    # negative semi-major axis
        "p kttsp 0 0.001 150 100 600 5\n1.5e7 0 0 0 0 0\n1.5e7 1.2 0 0 0 0\n",   # hyperbolic orbit
        "p kttsp 10 0.001 5 100 600 5\n1.5e7 0 0 0 0 0\n1.5e7 0 0 0 0 0\n",      # t_max before t0
        "p kttsp 0 -1 150 100 600 5\n1.5e7 0 0 0 0 0\n1.5e7 0 0 0 0 0\n",        # negative minimum flight time
        "p kttsp 0 0.001 150 100 50 5\n1.5e7 0 0 0 0 0\n1.5e7 0 0 0 0 0\n",      # exception limit below normal
        "p kttsp 0 0.001 150 100 600 5.5\n1.5e7 0 0 0 0 0\n1.5e7 0 0 0 0 0\n",   # fractional exception count
        "p kttsp 0 0.001 150 100 600 x\n1.5e7 0 0 0 0 0\n1.5e7 0 0 0 0 0\n",     # not a number
    ],
)
def test_malformed_files_are_rejected(text):
    with pytest.raises(ValueError):
        parse_instance(text)


def test_periods_and_speeds(easy):
    low, high = easy.groups()
    assert np.allclose(easy.period()[high], 1.906, atol=0.01)
    assert np.allclose(easy.period()[low], 0.238, atol=0.002)
    assert np.allclose(easy.v_circ()[high], 572.0, atol=2.0)
    assert np.allclose(easy.v_circ()[low], 1144.0, atol=3.0)


def test_groups_and_clusters(easy):
    low, high = easy.groups()
    v_low, v_high = 1144.0, 572.0
    # the low tomatoes form three inclination clusters that even an exception move cannot connect
    low_clusters = easy.clusters(low, easy.max_plane_change(easy.dv_exception, v_low))
    assert len(low_clusters) == 3
    assert sorted(len(c) for c in low_clusters) == [3, 3, 3]
    # the high tomatoes form one chain that normal moves can follow
    high_clusters = easy.clusters(high, easy.max_plane_change(easy.dv_max, v_high))
    assert len(high_clusters) == 1 and len(high_clusters[0]) == 40
    assert set(np.concatenate(low_clusters).tolist()) == set(low.tolist())


def test_plane_change_matrix(easy):
    rel = easy.relative_inclination_matrix()
    dv = easy.plane_change_matrix()
    assert np.allclose(np.diag(rel), 0.0, atol=1e-7)
    assert np.allclose(rel, rel.T)
    assert np.all(dv >= 0.0) and np.all(dv <= 2.0 * easy.v_circ()[:, None] + 1e-9)
    # a pure plane change of 0.175 rad at 572 m/s costs about 100 m/s
    assert easy.max_plane_change(100.0, 572.0) == pytest.approx(0.1752, abs=1e-3)


def test_h_z_sign(easy):
    s = easy.h_z_sign()
    assert set(s.tolist()) <= {-1, 1}
    assert np.all(s[easy.inc < 1.0] == 1) and np.all(s[easy.inc > 2.0] == -1)
