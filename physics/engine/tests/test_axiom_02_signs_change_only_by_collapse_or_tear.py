"""Axiom 02: plus and minus never cancel; only joint collapse turns minus into plus, only a tear turns plus into minus.

Bible rule 4. Default universe (seed 1, Settings(), axiom_run.py), 300 steps. After every
step: every box whose sign turned minus to plus carries the collapse mark, every box
whose sign turned plus to minus carries the split mark (a box that changed cell, emptied
and refilled by its new cell in one step, is counted apart and printed: its quanta did
not turn), and the plus total changed by
exactly the quanta collapse turned plus minus the quanta tears turned minus (plus and
minus never cancel: no other rule changes the plus total). Staged trigger: the 8-volume
even start of axiom 10 (new_universe(5, 10, seed=1, unevenness=0, margin=8)) with joint
J between volumes (2, 2, 2) and (4, 2, 2) left with 100 quanta per box, so it wants a
thousand times its room and is squeezed past the collapse limit; one full step():
every box that turned minus to plus is marked as collapse, none turned plus to minus,
and the plus gain equals the collapsed quanta, the amount exact. A box that changed cell
in the same step (emptied and refilled by its new cell, owner-change mark 2) is counted
apart as in the default run: its quanta did not turn.
"""
import numpy as np

from physics.engine import Settings, State, host_boxes, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu

J = (3 * 7 + 2) * 7 + 2


@needs_gpu
def test_every_sign_change_in_the_default_run_is_a_collapse_or_a_tear():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    previous_plus, previous_collapsed, previous_torn = run["start"]["plus"], 0, 0
    unmarked, unexplained, flips = 0, 0, np.zeros(6, dtype=np.int64)
    for row in run["steps"]:
        flips += row["flips"]
        unmarked += row["flips"][2] + row["flips"][5]
        expected = (row["tallies"][3] - previous_collapsed) - (row["torn"] - previous_torn)
        unexplained += abs((row["plus"] - previous_plus) - expected)
        previous_plus, previous_collapsed, previous_torn = row["plus"], row["tallies"][3], row["torn"]
    print(f"300 steps: boxes minus->plus by collapse {flips[0]}, by rehoming (emptied, refilled) {flips[1]}, "
          f"by nothing marked {flips[2]}; plus->minus by tear {flips[3]}, by rehoming {flips[4]}, by nothing "
          f"marked {flips[5]}; plus-total change not explained by collapse or tear: {unexplained} quanta")
    assert unmarked == 0 and unexplained == 0


@needs_gpu
def test_a_staged_collapse_is_the_only_minus_to_plus_and_keeps_the_amount():
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == J] = -100
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    state = State.from_host(quanta, np.zeros(quanta.shape), seed=1, lattice=graph)
    before = totals(state)
    step(state, Settings())
    after = totals(state)
    now, marks = state.quanta.numpy(), state.marks.numpy()
    turned_plus = (quanta < 0) & (now > 0)
    turned_minus = (quanta > 0) & (now < 0)
    collapsed = int(state.tallies.numpy()[0, 3])
    collapsed_boxes = int((turned_plus & ((marks & 1) == 1)).sum())
    unmarked = int((turned_plus & ((marks & 3) == 0)).sum())
    print(f"minus->plus {int(turned_plus.sum())} boxes: collapse-marked {collapsed_boxes}, changed cell "
          f"{int(turned_plus.sum()) - collapsed_boxes - unmarked}, neither {unmarked}; "
          f"plus->minus {int(turned_minus.sum())}; plus gain {after['plus'] - before['plus']}, collapsed quanta "
          f"{collapsed}; amount {before['amount']} -> {after['amount']}")
    assert collapsed_boxes > 0 and unmarked == 0
    assert turned_minus.sum() == 0
    assert after["plus"] - before["plus"] == collapsed and after["amount"] == before["amount"]
