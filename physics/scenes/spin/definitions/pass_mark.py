"""Pass mark of the spin scene, written 2026-11-17 before any run.

Known: a whirl turns only in whole steps and never unwinds; spin does not change by itself.
Contract: `passes(result)` takes the scene's result (16 places, natural start, every
rule on, 2,000 steps) and returns True when every watched cell read its own whole turn
in every layer at the start (`born` all `expected`) and at every step
(`held_steps == steps`). `NOT_ASKED` names what the known answer includes and this scene
cannot ask; a scorecard counts each as a miss.
Restated 2026-11-18 for the rebuilt engine, before any run on it: the cells watched are
the start's own swirls, volumes and joints (every cell is born one whole swirl), where
the old mark watched four whirls wound in by hand; the placed double whirl is gone.
"""
from __future__ import annotations

NOT_ASKED = ("spin one half: a whirl turned once against its surroundings reads changed, turned twice the same "
             "(the belt trick)",)


def passes(result: dict) -> bool:
    """Whether every watched cell was one whole swirl at the start and at every step."""
    return bool(result["cases"]) and all(
        case["born"] == [case["expected"]] * len(case["born"]) and case["held_steps"] == result["steps"]
        for case in result["cases"])
