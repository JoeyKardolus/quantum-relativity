"""Axiom 2, Cells are swirls: plus only pulls, minus only pushes, both whirl in whole steps.

Staged:
  - plus only pulls: after one step from no pull, every energy volume lies below every
    joint in the pull and all joints are level (only plus is a source); a joint's own
    energy makes its squeeze smaller, never larger (minus only pushes), a volume's
    energy makes it larger
  - whirls turn in whole steps: one whole turn of phase placed round the middle of an
    energy volume's block, in a universe with every rule on, is still exactly one whole
    turn round the block's rim at every one of 1,000 steps
    (physics.measure.phase_winding), and the cell's own boxes hold its quanta throughout
  - the same in a joint (at the start's 100 quanta a box; at 30 a box the rounding to
    whole quanta lost the whirl by step 268, which is why joints start with 100)
"""
import numpy as np

from physics.engine import BLOCK, ENERGY_VOLUME, JOINT, Settings, cell_laws, new_universe, step
from physics.engine.squeeze import squeeze
from physics.engine.tests._gpu import needs_gpu
from physics.measure import phase_winding


@needs_gpu
def test_only_plus_is_a_source_of_pull():
    state = new_universe(8)
    step(state, Settings())
    pull, kind = state.host("pull"), state.kind_h[:state.count]
    joints, volumes = pull[kind == JOINT], pull[kind == ENERGY_VOLUME]
    print(f"after one step: pull of the joints {joints.min():.3e} .. {joints.max():.3e}, of the energy volumes "
          f"{volumes.min():.3e} .. {volumes.max():.3e}")
    assert joints.max() - joints.min() < 1e-9 and volumes.max() < joints.min()


@needs_gpu
def test_a_volumes_energy_squeezes_a_joint_and_the_joints_own_energy_pushes_back():
    joint, low = (1 * 8 + 2) * 8 + 2, (0 * 8 + 2) * 8 + 2
    found = {}
    for name, cell, amount in (("start", None, 0), ("volume doubled", low, 2 * 648000), ("joint doubled", joint, 2 * 21600)):
        state = new_universe(8, unevenness=0.0, inner_unevenness=0.0)
        if cell is not None:
            state.fill(cell, amount)
        cell_laws(state, Settings())
        squeeze(state, Settings(squeeze_rate=0.0))
        found[name] = float(state.host("squeeze")[joint])
    print(f"squeeze of one joint: {found}")
    assert found["volume doubled"] > found["start"] > found["joint doubled"] > 0.0
    assert found["start"] / found["joint doubled"] > 8.0


def _whirl() -> np.ndarray:
    """One whole turn of phase round the middle line (along the third axis) of a block."""
    spot = np.arange(BLOCK) - (BLOCK - 1) / 2.0
    first, second = np.meshgrid(spot, spot, indexing="ij")
    return np.repeat(np.arctan2(second, first)[:, :, None], BLOCK, axis=2).astype(np.float32)


def _rim_turns(quanta: np.ndarray, phase: np.ndarray) -> list:
    """The whole turns of phase round the rim of each layer of a block, by summing the unit squares inside."""
    field = np.sqrt(np.abs(quanta).astype(np.float64)) * np.exp(1j * phase.astype(np.float64))
    return phase_winding(field)[2].sum(axis=(0, 1)).tolist()


def _whirl_run(cell: int, steps: int, **made) -> tuple:
    """Place one whole turn in `cell`, run with every rule on; returns (rim turns seen, first step not all 1, broken looks)."""
    state, settings = new_universe(8, **made), Settings()
    state.set_block(cell, state.block(cell)[0], _whirl())
    seen, first, broken = set(), None, 0
    for count in range(1, steps + 1):
        step(state, settings)
        quanta, phase = state.block(cell)
        turns = set(_rim_turns(quanta, phase))
        seen |= turns
        if first is None and turns != {1}:
            first = count
        broken += int(np.abs(quanta).sum() != state.host("cell_quanta")[cell]) + int((quanta == 0).any())
    return seen, first, broken


@needs_gpu
def test_a_whirl_in_an_energy_volume_stays_one_whole_turn_for_1000_steps():
    seen, first, broken = _whirl_run((2 * 8 + 2) * 8 + 2, 1000)
    print(f"energy volume, 1,000 steps, every rule on: whole turns round the rim of every layer, all values seen: "
          f"{seen}; looks at which the cell's boxes did not hold its quanta: {broken}")
    assert seen == {1} and first is None and broken == 0


@needs_gpu
def test_a_whirl_in_a_joint_stays_one_whole_turn_for_1000_steps():
    seen, first, broken = _whirl_run((2 * 8 + 3) * 8 + 2, 1000)
    print(f"joint, 1,000 steps, every rule on: whole turns round the rim of every layer, all values seen: {seen}; "
          f"looks at which the cell's boxes did not hold its quanta: {broken}")
    assert seen == {1} and first is None and broken == 0
