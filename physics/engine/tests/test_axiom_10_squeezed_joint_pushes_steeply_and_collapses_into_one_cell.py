"""Axiom 10: a joint is squeezed when smaller than its energy wants; push rises steeply; past the limit it collapses into ONE cell at once.

Audit answers (owner, 2026-11-16): a joint is squeezed when it is smaller than its energy
wants; when a joint collapses, its two volumes become one cell at once. Bible rule 8:
push rises steeply, like a hard wall. Staged on new_universe(5, 10, seed=1,
unevenness=0, margin=8) (8 energy volumes, every box 1000, at rest), joint J at lattice
(3, 2, 2) between energy volumes (2, 2, 2) and (4, 2, 2), Settings():
  1. steep push: J's quanta scaled so its wanted size is 1, 2, 4 and 8 times its size
     (wanted side = size_energy / quanta, bible rule 4, the joints' constant
     state.size_energy[0, 1]); the engine's own cell push (axiom_pressure.py) rises with
     that ratio at least as its cube at every step up
     (steeper than a gas pressed at fixed temperature, which rises as the ratio)
  2. collapse: J left with 100 quanta per box (a tenth of the start's), so it wants a
     side 10 times longer than it has, a thousand times its room: squeezed far past any
     limit. Within 30 steps J collapses, and at the step it does the lattice holds 7
     energy volumes (its two became one cell) and J holds no minus quanta
"""
import numpy as np

from physics.engine import Settings, State, cells, host_boxes, new_universe, stats, step
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_pressure import axiom_pressure

J = (3 * 7 + 2) * 7 + 2


def _start(joint_packing: int) -> State:
    """The 8-volume even start with J at `joint_packing` minus quanta per box."""
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == J] = -joint_packing
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    return State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=graph)


@needs_gpu
def test_a_joint_smaller_than_its_energy_wants_pushes_steeply():
    settings = Settings()
    ratios, pushes = [], []
    for wanted in (1, 2, 4, 8):
        state = _start(int(round(1000 * wanted ** (-1.0 / 3.0))))
        pressure, size, amount = axiom_pressure(state, settings)
        ratios.append((state.size_energy[0, 1] / amount[J]) ** 3 / size[J])
        pushes.append(pressure[J])
    slopes = np.diff(np.log(pushes)) / np.diff(np.log(ratios))
    print("wanted/size " + " ".join(f"{value:.3f}" for value in ratios) + "; push "
          + " ".join(f"{value:.4g}" for value in pushes) + "; log slope " + " ".join(f"{value:.2f}" for value in slopes))
    assert np.all(slopes >= 3.0)


@needs_gpu
def test_a_joint_squeezed_past_the_limit_collapses_and_its_two_volumes_become_one_cell():
    settings = Settings()
    state = _start(100)
    first, volumes_then, minus_left = None, None, None
    for count in range(1, 31):
        step(state, settings)
        if first is None and int(state.tallies.numpy()[0, 1]) > 0:
            first = count
            found = cells(host_boxes(state))
            volumes_then = found["volumes"]
            minus_left = int(-host_boxes(state)["quanta"][host_boxes(state)["owner"] == J].clip(max=0).sum())
    measured = stats(state, settings)
    print(f"J at 100 per box: first collapse at step {first}; energy volumes then {volumes_then}; J's minus quanta "
          f"then {minus_left}; after 30 steps collapses {measured['collapses']}, energy volumes {measured['cells']}, "
          f"squeeze by packing max {measured['squeeze_max']:.3f}")
    assert first is not None
    assert volumes_then == 7 and minus_left == 0
