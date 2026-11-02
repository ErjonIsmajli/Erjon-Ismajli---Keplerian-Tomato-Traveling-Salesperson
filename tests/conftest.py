from pathlib import Path

import numpy as np
import pytest

from kttsp.instance import Instance, load_instance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def make_instance(elements, name="synthetic", t0=0.0, min_tof=0.001, t_max=100.0,
                  dv_max=100.0, dv_exception=600.0, n_exceptions=5) -> Instance:
    """Build a small Instance from a list of element rows (a, e, i, raan, argp, M0)."""
    return Instance(name, t0, min_tof, t_max, dv_max, dv_exception, n_exceptions,
                    np.asarray(elements, dtype=float))


@pytest.fixture(scope="session")
def easy() -> Instance:
    return load_instance(DATA / "easy.kttsp")


@pytest.fixture(scope="session")
def medium() -> Instance:
    return load_instance(DATA / "medium.kttsp")


@pytest.fixture(scope="session")
def hard() -> Instance:
    return load_instance(DATA / "hard.kttsp")
