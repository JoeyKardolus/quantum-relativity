"""Axiom 03: every energy volume is wrapped by joints on all 26 sides, in the boxes.

Bible "The foundation" 1 (owner, 2026-11-16: never touching on a face, an edge or a
corner). Default universe (seed 1, Settings(), axiom_run.py), 300 steps. After every
step: 0 box pairs of two different energy volumes are 26-neighbours, and 0 plus boxes
have an empty box or a wall among their 26 neighbours (axiom_contacts.py).
"""
from physics.engine.tests._gpu import needs_gpu


@needs_gpu
def test_no_two_volumes_touch_and_no_volume_touches_empty_space_at_any_step():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    touching = max(row["touching"] for row in run["steps"])
    opened = max(row["open_boxes"] for row in run["steps"])
    print(f"300 steps: most touching volume box pairs at one step {touching}; most plus boxes beside empty "
          f"space {opened}")
    assert len(run["steps"]) >= 300
    assert touching == 0 and opened == 0
