"""Tracers across: a tracer in a volume's box that faces a joint is carried across with the fluid.

The engine's own passage says how much crosses between the two facing boxes; the tracer
keeps as much of their fluid behind it as it had, so it goes over when the fluid does
and lands in the other volume. A joint's own tracers never leave it.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, wave_of
from physics.engine.axiom_4_crossing.definitions.passage import passing_turn, row_minus, rub_share
from physics.view.definitions.carried import carried


@wp.kernel
def _carry_across(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
                  slot: wp.int32, cells: wp.int32, across: wp.int32, row: wp.int32,
                  tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[row, tracer]
    cell = wp.clamp(int(place[1]) / across, 0, cells - 1)
    line = place[1] - float(cell * across + across - 1)  # along the pair, were it the left volume
    left = cell
    if line < 0.0:  # not in its cell's last box: in its first, the pair's right volume?
        left = cell - 2
        line += float(across)
    if cell % 2 == 1 or left < 0 or left + 2 >= cells or line < 0.0 or line >= 2.0:
        return
    first = wp.clamp(int(place[0]), 0, across - 1) * across * across + wp.clamp(int(place[2]), 0,
                                                                                across - 1)
    minus = row_minus(quanta, slot + left + 1, first, across)
    if minus <= wp.float64(0.0):
        return
    low, high = slot + left, slot + left + 2
    low_has, high_has = quanta[low, first + (across - 1) * across], quanta[high, first]
    _low_wave, high_wave = pair_rotation(
        wave_of(low_has, phase[low, first + (across - 1) * across]),
        wave_of(high_has, phase[high, first]), passing_turn(rub_share(minus, low_has, high_has)))
    stays = float(low_has + high_has) - float(wp.dot(high_wave, high_wave))
    line = carried(line, float(low_has), float(high_has), stays)
    if line < 1.0:
        place[1] = float(left * across + across - 1) + line  # still in the left volume
    else:
        place[1] = float((left + 2) * across) + line - 1.0  # over, in the right volume
    tail[row, tracer] = place


def carry_across(box_quanta: wp.array, box_phase: wp.array, slot: int, cells: int, across: int,
                 tail: wp.array, row: int) -> None:
    """Carry the tracers of row `row` across the joints of `cells` cells from `slot` on."""
    if cells > 2:
        wp.launch(_carry_across, dim=tail.shape[1], device=tail.device,
                  inputs=[box_quanta, box_phase, slot, cells, across, row, tail])
