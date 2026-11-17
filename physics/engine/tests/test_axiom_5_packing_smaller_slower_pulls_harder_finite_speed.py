"""Axiom 5, Packing: more packed is smaller, slower and pulls harder; pull spreads at a finite speed, weaker with distance.

Checked at every step of the default run (1,000 steps), axiom_run.py: every living
cell's clock moved on by step_time times its own rate.
Staged (a level universe, 16 places a side, crossing and squeeze off so nothing but the
pull moves; one energy volume packed to twice the energy, beside the same universe
without it):
  - smaller: the packed volume is exactly half as long; slower: its clock runs slower
    and after 200 steps has counted less time
  - pull at a finite speed: the difference the packed volume makes is exactly level
    (to rounding) at every cell more neighbour-steps away than steps taken, for the
    first 6 steps; and at the volumes 1 to 4 places along a row it reaches half its
    deepest no sooner than light takes over the sizes of the cells between, by the
    shortest way (which runs through the thin joints of the web, not through the
    volumes); when a tenth arrives is printed too. Below that a fore-runner moves one
    neighbour a step, which inside a big cell is faster than light: a known miss
  - pulls harder, weaker with distance: averaged over the last 2,000 of 4,000 steps the
    pull is deepest at the packed volume and less deep at each volume further along the
    row; packed to three times the energy it is deeper still
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from physics.engine import ENERGY_VOLUME, Settings, new_universe, step
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_run import axiom_run

SIDE = 16
HOME = (8 * SIDE + 8) * SIDE + 8
QUIET = dict(crossing_rate=0.0, squeeze_rate=0.0, tear_length=0.0)


def _level(times: float):
    state = new_universe(SIDE, unevenness=0.0, inner_unevenness=0.0)
    if times != 1.0:
        state.fill(HOME, int(times * 648000))
    return state


def _hops() -> np.ndarray:
    """Neighbour-steps from HOME to every lattice place, round the edge."""
    places = np.arange(SIDE ** 3)
    index = np.stack([places // (SIDE * SIDE), (places // SIDE) % SIDE, places % SIDE], axis=1)
    away = np.abs(index - 8)
    return np.minimum(away, SIDE - away).max(axis=1)


@needs_gpu
def test_every_clock_ticks_at_its_own_rate_at_every_step():
    steps = axiom_run("default")["steps"]
    wrong = sum(row["clock_wrong"] for row in steps)
    print(f"default: cells whose clock did not move on by step_time * rate, summed over {len(steps)} steps: {wrong}")
    assert wrong == 0


@needs_gpu
def test_a_packed_volume_is_smaller_and_its_clock_runs_slower():
    state, settings = _level(2.0), Settings(**QUIET)
    for _ in range(200):
        step(state, settings)
    length, rate, clock = state.host("length"), state.host("rate"), state.host("clock")
    other = HOME + 2
    print(f"twice the energy: length {length[HOME]:.4f} against {length[other]:.4f}, clock rate {rate[HOME]:.4f} "
          f"against {rate[other]:.4f}, time lived in 200 steps {clock[HOME]:.3f} against {clock[other]:.3f}")
    assert length[HOME] == np.float32(length[other] / 2.0)
    assert rate[HOME] < rate[other] and clock[HOME] < clock[other]
    assert abs(clock[other] - 200 * settings.step_time * rate[other]) < 1e-2


@needs_gpu
def test_the_pull_spreads_one_neighbour_a_step_at_most_and_never_faster_than_light():
    settings = Settings(**QUIET)
    packed, plain = _level(2.0), _level(1.0)
    hops = _hops()
    row = [HOME + 2 * places for places in range(1, 5)]
    leak, trace = [], []
    for count in range(1, 1601):
        step(packed, settings)
        step(plain, settings)
        change = (packed.host("pull")[:SIDE ** 3] - plain.host("pull")[:SIDE ** 3]).astype(np.float64)
        if count <= 6:
            beyond, inside = change[hops > count], change[hops <= count]
            leak.append(float(np.ptp(beyond)) / float(np.ptp(inside)))
        trace.append([change[cell] - change[0] for cell in row])
    trace = np.array(trace)
    light = [(reach - 0.5 * near - 0.5 * far_end) / (settings.light_speed * settings.step_time)
             for reach, near, far_end in _between(plain, row)]
    half = [int(np.flatnonzero(trace[:, places] <= 0.5 * trace[:, places].min())[0]) + 1 for places in range(4)]
    tenth = [int(np.flatnonzero(trace[:, places] <= 0.1 * trace[:, places].min())[0]) + 1 for places in range(4)]
    print(f"spread of the difference beyond the reach of n steps over the spread inside, n = 1..6: "
          f"{[f'{value:.1e}' for value in leak]}")
    print(f"volumes 1..4 places along the row: light over the sizes of the cells between (shortest way, through the "
          f"joint web) takes {[round(value, 1) for value in light]} steps; the pull reaches half its deepest at steps "
          f"{half}, a tenth at steps {tenth}")
    assert max(leak) < 1e-3
    assert all(arrival >= needed for arrival, needed in zip(half, light))


def _between(state, row: list) -> list:
    """Per cell of `row`: (shortest sum of sizes from HOME's middle to its middle, HOME's length, its length)."""
    length = state.host("length").astype(np.float64)
    rows, cols, reach = [], [], []
    for cell in range(SIDE ** 3):
        ids = state.neighbours(cell)[0]
        rows.extend([cell] * len(ids))
        cols.extend(ids.tolist())
        reach.extend((0.5 * (length[cell] + length[ids])).tolist())
    away = dijkstra(coo_matrix((reach, (rows, cols)), shape=(SIDE ** 3, SIDE ** 3)).tocsr(), indices=HOME)
    return [(float(away[cell]), float(length[HOME]), float(length[cell])) for cell in row]


@needs_gpu
def test_the_pull_is_deepest_at_the_packed_volume_weaker_with_distance_and_deeper_when_more_packed():
    settings = Settings(**QUIET)
    row = [HOME + 2 * places for places in range(0, 5)]
    far = (0 * SIDE + 0) * SIDE + 0
    depth = {}
    for times in (2.0, 3.0):
        state = _level(times)
        mean = np.zeros(SIDE ** 3)
        for count in range(4000):
            step(state, settings)
            if count >= 2000:
                mean += state.host("pull")[:SIDE ** 3] / 2000.0
        depth[times] = [float(mean[cell] - mean[far]) for cell in row]
        volumes = state.kind_h[:SIDE ** 3] == ENERGY_VOLUME
        assert int(np.flatnonzero(volumes)[np.argmin(mean[volumes])]) == HOME
    print(f"mean pull below the far corner at the packed volume and 1..4 places along its row: twice the energy "
          f"{[f'{value:.3e}' for value in depth[2.0]]}, three times {[f'{value:.3e}' for value in depth[3.0]]}")
    assert all(later > earlier for earlier, later in zip(depth[2.0], depth[2.0][1:])) and depth[2.0][0] < 0
    assert depth[3.0][0] < depth[2.0][0]
