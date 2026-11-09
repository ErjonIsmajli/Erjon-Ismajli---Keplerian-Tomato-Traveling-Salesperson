"""End-to-end checks of the command line interface on a small synthetic instance file."""
import json

import pytest

from kttsp.cli import build_parser, main

TINY = """c four high tomatoes on an inclination chain and a low cluster of two
p kttsp 0.0 0.001 60.0 100.0 600.0 2
1.5000000e+07 3.0e-03 0.00 0.000 0.6 1.2
1.5030000e+07 3.0e-03 0.08 0.002 2.1 4.0
1.5060000e+07 3.0e-03 0.16 0.004 5.0 0.3
1.5090000e+07 3.0e-03 0.24 0.006 3.3 2.2
3.7500000e+06 2.0e-03 0.01 0.001 1.1 5.5
3.7537500e+06 2.0e-03 0.05 0.001 4.4 0.9
"""


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli")
    inst = folder / "tiny.kttsp"
    inst.write_text(TINY)
    main(["solve", str(inst), "--starts", "1", "--lookahead", "0", "--workers", "4",
          "--out", str(folder / "tiny.json")])
    return folder, inst


def test_solve_writes_solution_and_submission(solved):
    folder, _ = solved
    sol = json.loads((folder / "tiny.json").read_text())
    sub = json.loads((folder / "submission_tiny.json").read_text())
    assert sub[0]["decisionVector"] == sol["chromosome"] and len(sol["chromosome"]) == 3 * 6 - 2


def test_evaluate_exit_codes(solved, capsys):
    folder, inst = solved
    with pytest.raises(SystemExit) as ok:
        main(["evaluate", str(inst), str(folder / "tiny.json")])
    assert ok.value.code == 0 and "feasible: True" in capsys.readouterr().out
    with pytest.raises(SystemExit) as strict:
        main(["evaluate", str(inst), str(folder / "tiny.json"), "--min-tof", "100"])
    assert strict.value.code == 2


def test_info_improve_and_plot(solved, capsys):
    folder, inst = solved
    main(["info", str(inst)])
    assert "low cluster 0: 2 tomatoes" in capsys.readouterr().out
    better = folder / "improved" / "better.json"          # written into a folder that does not exist yet
    main(["improve", str(inst), str(folder / "tiny.json"), "--budget", "0", "--out", str(better)])
    assert better.exists() and (better.parent / "submission_better.json").exists()
    main(["plot", str(inst), str(better), "--out", str(folder / "figs")])
    assert len(list((folder / "figs").glob("tiny_*.png"))) == 5


def test_readable_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as missing:
        main(["info", str(tmp_path / "missing.kttsp")])
    assert missing.value.code == 1 and "error" in capsys.readouterr().err
    bad = tmp_path / "bad.kttsp"
    bad.write_text("p kttsp 0 0.001 150 100 600 5\n1.5e7 1.5 0 0 0 0\n1.5e7 0 0 0 0 0\n")
    with pytest.raises(SystemExit) as invalid:
        main(["info", str(bad)])
    assert invalid.value.code == 1 and "eccentricity" in capsys.readouterr().err
    not_a_solution = tmp_path / "x.json"
    not_a_solution.write_text("{}")
    good = tmp_path / "tiny.kttsp"
    good.write_text(TINY)
    with pytest.raises(SystemExit) as wrong_file:
        main(["evaluate", str(good), str(not_a_solution)])
    assert wrong_file.value.code == 1 and "not a valid solution file" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["solve", "x.kttsp", "--budget", "-5"],
    ["solve", "x.kttsp", "--workers", "0"],
    ["improve", "x.kttsp", "s.json", "--chains", "0"],
    ["evaluate", "x.kttsp", "s.json", "--min-tof", "-1"],
])
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_documented_command_lines_parse():
    lines = [
        "solve data/easy.kttsp --workers 8 --budget 900 --chains 8 --no-start-high --out solutions/easy.json",
        "improve data/easy.kttsp solutions/easy.json --budget 0",
        "solve data/medium.kttsp --workers 6 --budget 1800 --lookahead 0 --out solutions/medium.json",
        "improve data/medium.kttsp solutions/medium.json --budget 600 --chains 6",
        "improve data/medium.kttsp solutions/medium.json --budget 0 --min-tof 0.01",
        "solve data/hard.kttsp --workers 6 --window 30 --candidates 8 --starts 2 --budget 3600 --lookahead 0 "
        "--out solutions/hard.json",
        "evaluate data/medium.kttsp solutions/medium.json --min-tof 0.01",
        "plot data/easy.kttsp runs/easy.json --out runs/figures",
    ]
    for line in lines:
        args = build_parser().parse_args(line.split())
        assert callable(args.func)
