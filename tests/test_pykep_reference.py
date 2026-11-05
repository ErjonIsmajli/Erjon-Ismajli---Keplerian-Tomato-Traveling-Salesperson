"""Compares our transfer costs with values computed by pykep (ESA's official library).
The fixture is produced by scripts/crosscheck_pykep.py --write-fixture inside the Docker image."""
import json
from pathlib import Path

import numpy as np
import pytest

from kttsp.instance import load_instance
from kttsp.kepler import Ephemeris
from kttsp.transfer import leg_dv
from tests.conftest import DATA

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pykep_reference.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="pykep reference fixture not generated")
@pytest.mark.parametrize("name", ["easy", "medium", "hard"])
def test_leg_dv_matches_pykep(name):
    ref = json.loads(FIXTURE.read_text())
    eph = Ephemeris(load_instance(DATA / f"{name}.kttsp"))
    rows = ref[name]["legs"]
    assert len(rows) >= 100
    for i, j, t_dep, tof, dv_ref in rows:
        dv = leg_dv(eph, int(i), int(j), float(t_dep), float(tof))
        if dv_ref is None:      # pykep found no arc at all
            assert dv == np.inf
            continue
        assert abs(dv - dv_ref) < 1e-6, (i, j, t_dep, tof, dv, dv_ref)
