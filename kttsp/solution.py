"""In-memory representation of a tour, conversion to the official decision vector and file formats."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHALLENGE_ID = "spoc-4-keplerian-tomato-traveling-salesperson"
PROBLEM_IDS = {"easy": "small", "medium": "medium", "hard": "large"}   # instance name -> problem id on the platform


class SolutionFileError(ValueError):
    """A file that is not a readable solution file."""


@dataclass
class Solution:
    order: np.ndarray                 # (N,) visiting sequence of tomato indices
    t_dep: np.ndarray                 # (N-1,) departure time of every leg [days]
    tof: np.ndarray                   # (N-1,) time of flight of every leg [days]
    instance: str = ""                # instance name (easy / medium / hard)
    dv: np.ndarray | None = None      # (N-1,) delta-v of every leg [m/s] when known
    objective: float | None = None    # mission time [days] when evaluated
    exceptions: list = field(default_factory=list)   # indices of the legs using the exception limit
    meta: dict = field(default_factory=dict)         # free-form information (solver settings, runtime)

    def __post_init__(self):
        self.order = np.asarray(self.order, dtype=int)
        self.t_dep = np.asarray(self.t_dep, dtype=float)
        self.tof = np.asarray(self.tof, dtype=float)
        if self.dv is not None:
            self.dv = np.asarray(self.dv, dtype=float)

    @property
    def n(self) -> int:
        return int(self.order.size)

    @property
    def t_end(self) -> float:
        return float(self.t_dep[-1] + self.tof[-1])

    # ---- official decision vector ----------------------------------------------------

    def to_chromosome(self) -> list:
        """x = [t_1..t_{N-1}, tof_1..tof_{N-1}, pi_1..pi_N] with plain Python numbers."""
        return [float(v) for v in self.t_dep] + [float(v) for v in self.tof] + [int(v) for v in self.order]

    @classmethod
    def from_chromosome(cls, x, n: int, instance: str = "") -> "Solution":
        x = np.asarray(x, dtype=float).ravel()
        if x.size != 3 * n - 2:
            raise ValueError(f"chromosome has {x.size} values, expected 3N-2 = {3 * n - 2}")
        m = n - 1
        return cls(np.rint(x[2 * m:]).astype(int), x[:m], x[m:2 * m], instance=instance)

    # ---- JSON files -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "instance": self.instance,
            "n_tomatoes": self.n,
            "objective_days": self.objective,
            "n_exceptions": len(self.exceptions),
            "exception_legs": [int(k) for k in self.exceptions],
            "order": [int(v) for v in self.order],
            "t_dep": [float(v) for v in self.t_dep],
            "tof": [float(v) for v in self.tof],
            "dv": None if self.dv is None else [float(v) for v in self.dv],
            "chromosome": self.to_chromosome(),
            "meta": self.meta,
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Solution":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(d["order"], d["t_dep"], d["tof"], instance=d.get("instance", ""), dv=d.get("dv"),
                       objective=d.get("objective_days"), exceptions=list(d.get("exception_legs", [])),
                       meta=dict(d.get("meta", {})))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SolutionFileError(f"{path} is not a valid solution file ({type(exc).__name__}: {exc})") from None

    def write_submission(self, path, problem_id: str | None = None, name: str = "", description: str = "") -> None:
        """Write the JSON file expected by the Optimise platform (same layout as its submission helper)."""
        if problem_id is None:
            problem_id = PROBLEM_IDS.get(self.instance, self.instance)
        entry = {"decisionVector": self.to_chromosome(), "problem": problem_id, "challenge": CHALLENGE_ID,
                 "name": name, "description": description}
        Path(path).write_text(json.dumps([entry], indent=6), encoding="utf-8")
