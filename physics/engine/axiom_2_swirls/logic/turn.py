"""Turn: energy is how fast a swirl turns. A cell's phase goes round in step with its quanta.

A calm volume's whirl turns by the step's own standing wave (`turn_per_step`). A cell that
holds more turns faster and one that holds less slower, in step with its quanta, as a
fluid that presses on itself does: each step every box of the cell is turned on by what
the cell's quanta add to (or take from) the calm turn. The whirl keeps its shape; only
how fast it goes round changes. The sign does not enter: a joint's whirl is the mirror of
a volume's, so the same turn carries it round the other way. `own_turned` keeps how far
each own cell has turned past the calm turn: a cell that turns faster is smaller by as
much (the one size law), so where two cells touch their fluid moves alike however their
energies differ, as a small gear meshes with a big one; the crossing reads it for that.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX


def turn_of(quanta: int, across: int, turn_per_step: float) -> float:
    """How far the whirl of a cell that holds `quanta` turns in a step: the calm turn times its quanta."""
    return turn_per_step * abs(quanta) / (QUANTA_A_BOX * across ** 3)


@wp.kernel
def _turn(own_cell: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
          calm: wp.float64, calm_turn: wp.float64, own_turned: wp.array(dtype=wp.float32),
          box_phase: wp.array2d(dtype=wp.float32)):
    slot, box = wp.tid()
    held = wp.abs(wp.float64(cell_quanta[own_cell[slot]]))
    more = calm_turn * (held / calm - wp.float64(1.0))  # the step itself gave the calm turn
    angle = wp.float64(box_phase[slot, box]) + more
    turn = wp.float64(wp.tau)
    box_phase[slot, box] = wp.float32(angle - turn * wp.floor((angle + wp.float64(wp.pi)) / turn))
    if box == 0:
        past = wp.float64(own_turned[slot]) + more
        own_turned[slot] = wp.float32(past - turn * wp.floor(past / turn))


def turn(store) -> None:
    """One step of every own cell's whirl turning by its quanta."""
    wp.launch(_turn, dim=(store.count, store.across ** 3), device=store.box_quanta.device,
              inputs=[store.own_cell, store.cell_quanta,
                      wp.float64(QUANTA_A_BOX * store.across ** 3),
                      wp.float64(store.turn_per_step), store.own_turned, store.box_phase])
