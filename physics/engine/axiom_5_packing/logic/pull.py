"""Pull: one step of every volume being pulled toward every other, by what the other held a delay ago.

On the card, a thread a volume (`definitions/pull.py` has the law, `pairs.py` reads it on
the host). First the cells' sizes are laid as ways (`ways.py`) and every body's energy is
written into the past (`groups.py`). Then each volume reads the others: the way to each,
how late its pull is, its energy then; near ones volume by volume, far ones group by group
(`definitions/groups.py`; `near` 0: every one by itself). What comes of it is written
twice: how deep the volume lies (`cell_pull`), and its fluid's flow, which gains its
quanta times the pull on it times the step, in whole numbers (nothing is cut off: the
crossing holds the speed under the top speed). A volume is in no sum of its own.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.spare import spare
from physics.engine.axiom_5_packing.definitions.groups import across_of, node_of, top_of, wide_of
from physics.engine.axiom_5_packing.definitions.pull import REACH, late_of
from physics.engine.axiom_5_packing.logic.groups import bodies
from physics.engine.axiom_5_packing.logic.pairs import past_of
from physics.engine.axiom_5_packing.logic.ways import middles, way_to
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

wp.set_module_options({"fuse_fp": False, "enable_backward": False})


@wp.kernel
def _pull(wide: wp.int32, top: wp.int32, near: wp.int32, step: wp.int32, calm_turn: wp.float64,
          a_step: wp.float64, middle: wp.array2d(dtype=wp.float64), energy: wp.array(dtype=wp.float64),
          stands: wp.array(dtype=wp.vec3d), lead: wp.array(dtype=wp.int32),
          past: wp.array2d(dtype=wp.float32), cell_quanta: wp.array(dtype=wp.int64),
          cell_pull: wp.array(dtype=wp.float32), drawn: wp.array(dtype=wp.vec3d),
          cell_flow: wp.array2d(dtype=wp.int64)):
    first, second, third = wp.tid()
    at = wp.vec3i(first, second, third)
    own, cell = node_of(wide, 0, at), lead[node_of(wide, 0, at)]
    deep, pulled = wp.float64(energy[own]), wp.vec3d()
    drawn[own] = pulled
    if deep <= wp.float64(0.0):
        return  # no volume stands here
    for level in range(top + 1):
        low, high = wp.vec3i(0, 0, 0), wp.vec3i(across_of(wide, level) - 1)
        if level < top:  # as far out as the near groups round its own group of the level above
            for axis in range(3):
                low[axis] = wp.max(2 * ((at[axis] >> (level + 1)) - near), 0)
                high[axis] = wp.min(2 * ((at[axis] >> (level + 1)) + near) + 1, high[axis])
        for one in range(low[0], high[0] + 1):
            for two in range(low[1], high[1] + 1):
                for three in range(low[2], high[2] + 1):
                    off = wp.max(wp.abs(one - (first >> level)), wp.max(
                        wp.abs(two - (second >> level)), wp.abs(three - (third >> level))))
                    other = node_of(wide, level, wp.vec3i(one, two, three))
                    if off == 0 or (level > 0 and off <= near) or energy[other] <= wp.float64(0.0):
                        continue  # itself, or read a level below, or nothing there
                    way = way_to(middle, 2 * wide - 1, cell, stands[other], lead[other])
                    apart = wp.length(way)
                    late = late_of(apart, a_step)
                    if late <= step:  # before a pull has had the time to arrive there is none
                        then = wp.float64(past[(step - late) % REACH, other])
                        deep += then / apart
                        pulled += then / (apart * apart * apart) * way
    cell_pull[cell] = wp.float32(deep)
    drawn[own] = pulled
    for axis in range(3):
        cell_flow[cell, axis] += wp.int64(wp.floor(
            wp.float64(cell_quanta[cell]) * pulled[axis] * calm_turn / wp.float64(wp.tau) + wp.float64(0.5)))


def pull(store: Store) -> None:
    """Write how deep every own volume lies and add a step of the pull on it to its flow."""
    if store.side == 0:
        return
    wide, top, a_turn = wide_of(store.side), top_of(store.side, store.near), store.turn_per_step / (2.0 * np.pi)
    past_of(store)
    energy, stands, lead = bodies(store, top, store.step % REACH)
    store.cell_pull.zero_()
    wp.launch(_pull, dim=(wide, wide, wide), device=store.box_quanta.device, inputs=[
        wide, top, store.near, store.step, wp.float64(store.turn_per_step), wp.float64(TOP_SPEED * a_turn),
        middles(store), energy, stands, lead, store.cell_past, store.cell_quanta, store.cell_pull,
        spare(store, "drawn", wide ** 3, wp.vec3d), store.cell_flow])
