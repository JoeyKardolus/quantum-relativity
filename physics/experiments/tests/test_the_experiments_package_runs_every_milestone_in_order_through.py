"""The experiments package runs every milestone in order through run_all.

WHAT IT PINS. physics/experiments through its public `run_all`: results come
back in milestone order 1..5, each with "plot_dir" pointing at the folder its
plots were written to, and milestone 2 (Grid dynamics, run for real here)
passes every rule check with its plot on disk.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The milestone
order is the bible's (validate, dynamics, gravity, expansion, space). The slow
experiments (1, 3, 4, 5) are swapped for stand-ins that write one real PNG
and return a minimal result, so only the ordering and folder plumbing of
run_all is judged for them; Grid dynamics runs unchanged, and its known values
come from the rule definitions ((1 + h)^n, conserved energy, unchanged count).

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. run_all recording out_dir instead of each
experiment's own folder as plot_dir, so the plots are not where it says.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from physics.experiments import expansion, gravity, run_all, space, validate  # noqa: E402


def _stand_in(milestone: int):
    def run(seed: int, out_dir: str) -> dict:
        figure, axes = plt.subplots(figsize=(1, 1))
        axes.plot([0, 1], [0, 1])
        figure.savefig(os.path.join(out_dir, "line.png"))
        plt.close(figure)
        return {"milestone": milestone, "verdict": "unclear", "plots": ["line.png"], "seed": seed}
    return run


def test_the_experiments_package_runs_every_milestone_in_order_through(tmp_path, monkeypatch) -> None:
    """Five results in milestone order, plots where plot_dir says, Grid dynamics passes."""
    for milestone, module in ((1, validate), (3, gravity), (4, expansion), (5, space)):
        monkeypatch.setattr(module, "run", _stand_in(milestone))
    results = run_all(7, str(tmp_path))
    assert [result["milestone"] for result in results] == [1, 2, 3, 4, 5]
    for result in results:
        assert result["seed"] == 7
        assert result["plot_dir"] != str(tmp_path)
        for name in result["plots"]:
            assert os.path.isfile(os.path.join(result["plot_dir"], name))
    dynamics = results[1]
    assert dynamics["verdict"] == "pass"
    for entry in dynamics["numbers"]:
        if entry["known"] is not None:
            assert abs(entry["measured"] - entry["known"]) < 1e-6, entry["name"]
