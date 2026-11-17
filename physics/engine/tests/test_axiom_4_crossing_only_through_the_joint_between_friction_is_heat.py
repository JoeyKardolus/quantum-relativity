"""Axiom 4, Crossing: energy diffuses between plus cells only through the minus between them; the only friction, kept as heat.

Checked at every step of the default run (1,000 steps), axiom_run.py:
  - what the volumes gained is what the volumes lost is what crossed the joints
  - a volume's quanta changed by exactly its gain minus its loss; no joint lost a
    quantum and no quantum is counted as crossing at a volume
Staged:
  - one packed volume gives only to the six volumes it shares a joint's way with
  - the crossing is steered by the pull: with equal quanta, energy flows to the volume
    deeper in the pull, and nothing flows between equal volumes at equal pull
  - the crossing stays plus and heats: the takers' own jiggle rises above the others'
"""
import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, new_universe, step, totals
from physics.engine.cross import cross
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_run import axiom_run


@needs_gpu
def test_what_crosses_leaves_one_volume_and_enters_another_through_a_joint_every_step():
    steps = axiom_run("default")["steps"]
    uneven = sum(1 for row in steps if not row["gained"] == row["lost"] == row["crossed_step"])
    crossed = sum(row["crossed_step"] for row in steps)
    print(f"default: quanta crossed in 1,000 steps {crossed}; steps where gained, lost and crossed differ: {uneven}; "
          f"volumes whose quanta did not change by gain - loss, summed: {sum(row['volume_wrong'] for row in steps)}; "
          f"quanta joints lost: {sum(row['joint_loss'] for row in steps)}; counted as crossing at a volume: "
          f"{sum(row['crossed_at_volumes'] for row in steps)}")
    assert crossed > 0 and uneven == 0
    assert sum(row["volume_wrong"] for row in steps) == 0
    assert sum(row["joint_loss"] for row in steps) == 0 and sum(row["crossed_at_volumes"] for row in steps) == 0
    assert steps[-1]["crossed"] == crossed


@needs_gpu
def test_a_packed_volume_gives_only_to_the_volumes_across_its_joints():
    state = new_universe(8, unevenness=0.0)
    settings = Settings(pull_strength=0.0)
    centre = (4 * 8 + 4) * 8 + 4
    state.fill(centre, 3 * 648000)
    start = totals(state)
    step(state, settings)
    gain, loss = state.host("gain"), state.host("loss")
    across = sorted(end for joint in state.joints_of(centre) for pair in state.ways(joint) for end in pair
                    if end != centre)
    takers = sorted(np.flatnonzero((gain > 100) & (state.kind_h[:state.count] == ENERGY_VOLUME)).tolist())
    after = totals(state)
    print(f"a volume of 3 times the energy lost {int(loss[centre])} quanta in one step; the volumes that took more "
          f"than 100: {takers}; the volumes across its joints: {across}; plus {start['plus']} -> {after['plus']}")
    assert takers == across and len(across) == 6 and loss[centre] > 0
    assert after["plus"] == start["plus"] and after["minus"] == start["minus"]


@needs_gpu
def test_the_crossing_is_steered_by_the_pull_toward_the_deeper_side():
    state = new_universe(8, unevenness=0.0, inner_unevenness=0.0)
    settings = Settings()
    near, far = (2 * 8 + 2) * 8 + 2, (4 * 8 + 2) * 8 + 2
    heat = np.full(state.capacity, 0.01, dtype=np.float32)
    state.jiggle.assign(heat)
    flows = {}
    for name, depth in (("level", 0.0), ("far deeper", -0.02), ("near deeper", 0.02)):
        pull = np.zeros(state.capacity, dtype=np.float32)
        pull[far] = depth
        state.pull.assign(pull)
        for array in (state.gain, state.loss, state.crossed):
            array.zero_()
        cross(state, settings)
        gain, loss = state.host("gain"), state.host("loss")
        flows[name] = int(gain[far] - loss[far]), int(gain[near] - loss[near]), int(loss.sum())
    print(f"equal volumes, net quanta into (far, near) and all quanta moved: {flows}")
    assert flows["level"] == (0, 0, 0)
    assert flows["far deeper"][0] > 0 and flows["near deeper"][0] < 0


@needs_gpu
def test_arrivals_turn_to_heat_and_stay_plus():
    state = new_universe(8, unevenness=0.0)
    settings = Settings(pull_strength=0.0)
    centre = (4 * 8 + 4) * 8 + 4
    state.fill(centre, 3 * 648000)
    for _ in range(50):
        step(state, settings)
    heat = state.host("jiggle")
    volumes = state.kind_h[:state.count] == ENERGY_VOLUME
    across = sorted(end for joint in state.joints_of(centre) for pair in state.ways(joint) for end in pair
                    if end != centre)
    others = volumes.copy()
    others[across + [centre]] = False
    books = totals(state)
    print(f"50 steps: heat per quantum of the six takers {heat[across].mean():.5f}, of the far volumes "
          f"{heat[others].mean():.5f}; boxes of the wrong sign {books['wrong_boxes']}")
    assert heat[across].min() > heat[others].max() and books["wrong_boxes"] == 0
