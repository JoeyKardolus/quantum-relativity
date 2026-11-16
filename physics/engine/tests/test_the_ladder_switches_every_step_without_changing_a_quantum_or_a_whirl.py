"""The detail ladder kept beside the engine switches its blocks every step, keeps both totals exact and never codes a whirl.

Issue #3 acceptance: a ladder region switches between detail levels without changing
total energy or breaking whirls. A 48-box cube is a joint web (-1000 per box) holding,
in its middle, a 24-box block of plus fluid (+1000 per box) with a whole-step whirl
along z through its middle (phase the angle round the line, the dip of the whirl test),
and an empty margin of 8 boxes. The engine runs 40 steps with the ladder attached (every
rule on, pull 0, as the whirl test does): after every step the ladder's plus and minus
totals equal the engine's exactly, every block a whirl line pierces is FULL, and its
boxes read back from the ladder exactly as the engine holds them. The levels seen over
the run are printed; nothing here asks that a block go COARSE (that depends on how quiet
the run is, which is the physics' business).
"""
import numpy as np

from physics.engine import Settings, State, attach_ladder, step, totals
from physics.engine.tests._gpu import needs_gpu
from physics.ladder import Level, read_region, total_quanta

SIDE = 48


@needs_gpu
def test_ladder_totals_match_and_whirl_blocks_stay_full():
    settings = Settings(pull_strength=0.0)
    quanta = np.zeros((SIDE,) * 3, dtype=np.int32)
    quanta[8:40, 8:40, 8:40] = -1000
    places = np.arange(SIDE) - 23.5
    across_x, across_y = np.meshgrid(places, places, indexing="ij")
    healing = 1.0 / np.sqrt(2.0 * settings.volume_stiffness)
    dip = np.maximum(np.rint(1000.0 * np.tanh(np.hypot(across_x, across_y) / healing) ** 2), 1).astype(np.int32)
    quanta[12:36, 12:36, 12:36] = dip[12:36, 12:36, None]
    phase = np.arctan2(across_y, across_x)[..., None] * np.ones(SIDE)
    state = State.from_host(quanta, phase)
    store = attach_ladder(state)
    levels = set()
    for count in range(40):
        step(state, settings)
        now = totals(state)
        plus, minus = total_quanta(store)
        assert (plus, minus) == (now["plus"], now["minus"])
        whirl_blocks = store.level[2:4, 2:4, 1:5]
        assert np.all(whirl_blocks == Level.FULL)
        levels |= {Level(value).name for value in np.unique(store.level)}
    read_quanta, read_phase = read_region(store, (2, 2, 1), (4, 4, 5))
    assert np.array_equal(read_quanta, state.quanta.numpy()[16:32, 16:32, 8:40])
    assert np.array_equal(read_phase, state.phase.numpy()[16:32, 16:32, 8:40])
    print(f"levels seen: {sorted(levels)}; at the end {np.unique(store.level, return_counts=True)}")
