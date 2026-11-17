"""Cross: plus energy crosses from energy volume to energy volume only through the joint between them, bent by the pull; the only friction.

Contract: `cross(state, settings)` works, for every way (a, b) of every joint (way.py),
the whole plus quanta that cross from volume a to volume b this step (negative: b to a):
  flow = crossing_rate * openness * (B(x) * Qa - B(-x) * Qb) * step_time * rate
Qa, Qb the volumes' quanta, B(x) = x / (e ** x - 1) (B(0) = 1), x = (pull_b - pull_a) / T
with T the mean of the two volumes' own heat per quantum (jiggle.py) and `rate` the
joint's clock: diffusion and drift together (the Smoluchowski law in its exact two-cell
form), plain difference when the pulls are equal, bent toward the volume deeper in the
pull, so energy attracts energy (axioms 4 and 5). Equal pulls give x = 0 whatever the
heat. openness is 1 for a joint as fat as the start's mean joint or fatter, falls in a
straight line as it thins and is 0 from seal_thin times thinner on: packed lumps seal
themselves. The flow becomes floor(|flow| + uniform) whole quanta (the uniform a hash of
the joint, the way, the state's key and the step), at most the giving volume's quanta
above 1 a box shared over the ways it is an end of (state.ways_n), so no volume can be
asked for more than it holds. The quanta are added to the giver's loss and the taker's
gain (pour.py moves them between the two cells' boxes, as plus, and turns the arrival
to heat), to the joint's state.crossed and to state.books[1]; the joint's own quanta
are not touched and the total does not change. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.settings import Settings
from physics.engine.state import BOXES, JOINT, State
from physics.engine.way import way

wp.set_module_options({"fuse_fp": False})

_EXP_LIMIT = 700.0  # numerics: e**700 is near float64's top; beyond it B is 0 or |x| to the last digit


@wp.func
def _bernoulli(bend: wp.float64) -> wp.float64:
    """B(x) = x / (e**x - 1), with B(0) = 1."""
    if wp.abs(bend) < wp.float64(1.0e-6):
        return wp.float64(1.0) - bend * wp.float64(0.5)
    return bend / (wp.exp(bend) - wp.float64(1.0))


@wp.kernel
def _cross(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
           rate: wp.array(dtype=wp.float32), pull: wp.array(dtype=wp.float32), heat: wp.array(dtype=wp.float32),
           ways_n: wp.array(dtype=wp.int32), way_at: wp.array(dtype=wp.int32),
           way_pool: wp.array(dtype=wp.int32), side_n: wp.array(dtype=wp.int32),
             side: int, speed: wp.float64, start: wp.float64,
           seal: wp.float64, seed: int, gain: wp.array(dtype=wp.int64), loss: wp.array(dtype=wp.int64),
           crossed: wp.array(dtype=wp.int64), books: wp.array(dtype=wp.int64)):
    joint = wp.tid()
    if kind[joint] != -1:
        return
    count = ways_n[joint]
    if count == 0:
        return
    thin = wp.float64(cell_quanta[joint]) / start
    openness = wp.clamp((seal - thin) / (seal - wp.float64(1.0)), wp.float64(0.0), wp.float64(1.0))
    if openness <= wp.float64(0.0):
        return
    for index in range(count):
        ends = way(joint, index, side, way_at, side_n, ways_n, way_pool)
        if ends[0] == ends[1]:
            continue
        near = ends[0]
        far = ends[1]
        fall = wp.float64(pull[far] - pull[near])
        warmth = wp.float64(0.5 * (heat[near] + heat[far]))
        limit = wp.float64(_EXP_LIMIT)
        bend = wp.float64(0.0)
        if fall > wp.float64(0.0):
            bend = limit
        if fall < wp.float64(0.0):
            bend = -limit
        if warmth > wp.float64(0.0):
            bend = wp.clamp(fall / warmth, -limit, limit)
        flow = speed * openness * wp.float64(rate[joint]) * (
            _bernoulli(bend) * wp.float64(cell_quanta[near]) - _bernoulli(-bend) * wp.float64(cell_quanta[far]))
        giver = near
        taker = far
        if flow < wp.float64(0.0):
            giver = far
            taker = near
            flow = -flow
        spare = (cell_quanta[giver] - wp.int64(BOXES)) / wp.int64(wp.max(ways_n[giver], 1))
        rng = wp.rand_init(seed + index * 104729, joint)
        moved = wp.min(wp.int64(wp.floor(wp.min(flow, wp.float64(spare)) + wp.float64(wp.randf(rng)))), spare)
        if moved > wp.int64(0):
            wp.atomic_add(loss, giver, moved)
            wp.atomic_add(gain, taker, moved)
            wp.atomic_add(crossed, joint, moved)
            wp.atomic_add(books, 1, moved)


def cross(state: State, settings: Settings) -> None:
    """Work out the whole plus quanta crossing every joint's ways this step."""
    seed = (state.step * 64 + 61) ^ state.key
    wp.launch(_cross, dim=state.count, device=state.device,
              inputs=[state.kind, state.cell_quanta, state.rate, state.pull, state.jiggle, state.ways_n,
                      state.way_at, state.way_pool, state.side_n, state.side,
                      wp.float64(settings.crossing_rate * settings.step_time),
                      wp.float64(state.start_quanta[JOINT]), wp.float64(settings.seal_thin),
                      int(seed & 0x7FFFFFFF), state.gain, state.loss, state.crossed, state.books])
