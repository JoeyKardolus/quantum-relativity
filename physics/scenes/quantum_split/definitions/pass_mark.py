"""Pass mark of the quantum split scene, rewritten 2026-11-18 for the rebuilt engine, before any run of this scene.

Known: a split energy is found whole in one place, in each place as often as its share.
Contract: `SETTLINGS` is 200 and `WITHIN` 0.05. `passes(result)` takes the scene's
result (32 places, natural start, 20 steps, then a detector on every open split) and
returns True when at least SETTLINGS settlings were judged, every settling was whole
(`whole_share` 1.0) and the share found in the largest half's cell is within WITHIN of
the share expected there. `NOT_ASKED` names what the known answer includes and this
scene cannot ask of the engine; a scorecard counts each as a miss.
The mark of 2026-11-17 (50/50 in 0.40 .. 0.60, 70/30 in 0.61 .. 0.79, rejoined in step
0.97 or more and out of step 0.03 or less, far side's odds 0.21 .. 0.39 with and
without a near reaction) judged staged splits; the engine no longer has the doors that
staged them, and splits open by themselves at whatever shares the fluid gives.
"""
from __future__ import annotations

SETTLINGS = 200
WITHIN = 0.05
NOT_ASKED = ("two routes brought back together in step give all to one exit, out of step none",
             "a reaction on the near side does not change the far side's odds (no message)",
             "Bell test: filter angles on pairs of whirls, 75% match against 50% for anything local")


def passes(result: dict) -> bool:
    """Whether enough splits settled, all whole, in their largest half as often as its share."""
    enough = result["judged"] >= SETTLINGS and result["whole_share"] == 1.0
    return bool(enough and abs(result["found_in_largest"] - result["expected_in_largest"]) <= WITHIN)
