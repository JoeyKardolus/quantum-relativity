"""Axiom 1, Energy: an always-moving fluid of whole signed quanta; the total is not fixed, and its books are exact.

The owner's axiom (bible physics/quantum-relativity, "The axioms", corrected 2026-11-17):
one whole signed number of quanta per Planck box; the total is not fixed: a squeezed
joint makes more minus energy out of itself, a collapse takes the joint's energy off the
rest. Checked at every step of the default run (1,000 steps) and of the eventful run
(60 steps through hundreds of tears and collapses), axiom_run.py:
  - every box of a living cell holds a whole number with its cell's sign, never 0, and
    every box of an id that is no cell holds 0
  - the books, exactly: amount = start + made - taken with the joint - taken off the
    volumes; plus = start - taken off the volumes - torn; minus = start + made - taken
    with the joint + torn
  - always moving: the boxes of 8 sample cells differ from the step before
Staged: a joint squeezed by two packed volumes makes minus quanta out of itself while
no volume loses one.
"""
import numpy as np

from physics.engine import JOINT, Settings, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_run import axiom_run


def _gaps(run: dict) -> tuple:
    start = run["start"]
    amount = [row["amount"] - (start["amount"] + row["made"] - row["taken_joint"] - row["taken_volumes"])
              for row in run["steps"]]
    plus = [row["plus"] - (start["plus"] - row["taken_volumes"] - row["torn"]) for row in run["steps"]]
    minus = [row["minus"] - (start["minus"] + row["made"] - row["taken_joint"] + row["torn"]) for row in run["steps"]]
    return amount, plus, minus


@needs_gpu
def test_the_books_are_exact_at_every_step_of_both_runs():
    for name in ("default", "eventful"):
        run = axiom_run(name)
        amount, plus, minus = _gaps(run)
        last = run["steps"][-1]
        print(f"{name}: {len(run['steps'])} steps, amount {run['start']['amount']} -> {last['amount']}, made by squeeze "
              f"{last['made']}, taken with collapsed joints {last['taken_joint']}, taken off their volumes "
              f"{last['taken_volumes']}, torn plus to minus {last['torn']}; largest gap in the books: amount "
              f"{max(map(abs, amount))}, plus {max(map(abs, plus))}, minus {max(map(abs, minus))}")
        assert not any(amount) and not any(plus) and not any(minus)
    eventful = axiom_run("eventful")["steps"][-1]
    assert eventful["made"] > 0 and eventful["taken_joint"] > 0 and eventful["taken_volumes"] > 0
    assert eventful["amount"] != axiom_run("eventful")["start"]["amount"]


@needs_gpu
def test_every_box_holds_a_whole_number_with_its_cells_sign_at_every_step():
    for name in ("default", "eventful"):
        run = axiom_run(name)
        wrong = sum(row["wrong_boxes"] for row in run["steps"])
        print(f"{name}: boxes that are 0, of the wrong sign or left in a dead id, summed over all steps: {wrong}")
        assert wrong == 0
    state = axiom_run("default")["state"]
    assert state.quanta.numpy().dtype == np.int32


@needs_gpu
def test_the_fluid_is_always_moving():
    steps = axiom_run("default")["steps"]
    still = sum(row["still"] for row in steps)
    print(f"default: sample cells whose boxes stood still for a step, of {8 * len(steps)} looks: {still}")
    assert still == 0


@needs_gpu
def test_a_squeezed_joint_makes_minus_energy_out_of_itself_and_takes_none_from_a_volume():
    state = new_universe(8, unevenness=0.0)
    settings = Settings(crossing_rate=0.0)
    joint, low, high = (1 * 8 + 2) * 8 + 2, (0 * 8 + 2) * 8 + 2, (2 * 8 + 2) * 8 + 2
    for volume in (low, high):
        state.fill(volume, 2 * 648000)
    start = totals(state)
    held = state.host("cell_quanta").copy()
    rises = []
    for _ in range(40):
        step(state, settings)
        rises.append(int(state.host("cell_quanta")[joint]))
    now, books = state.host("cell_quanta"), totals(state)
    volumes = state.kind_h[:state.count] == 1
    print(f"joint between two volumes of twice the energy: {int(held[joint])} -> {rises[-1]} quanta in 40 steps "
          f"(squeeze now {float(state.host('squeeze')[joint]):.3f}); volumes' quanta changed: "
          f"{int(np.abs(now[volumes] - held[volumes]).sum())}; total {start['amount']} -> {books['amount']}, made {books['made']}")
    assert rises[-1] > held[joint] and all(later >= earlier for earlier, later in zip(rises, rises[1:]))
    assert np.array_equal(now[volumes], held[volumes])
    assert books["amount"] - start["amount"] == books["made"] > 0
    assert books["minus"] - start["minus"] == books["made"] and state.kind_h[joint] == JOINT
