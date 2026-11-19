"""Line streaks: the tails of a scene's cells as the card draws them, each cell in its shape.

A tracer's second number says which slot's cell it is in (`across` a cell); `shape.py`
lays the cell's block out where and as wide as `placed` says.

One rule for both colours: tracers are laid by the quanta, so each stands for as much
energy, and every mark is drawn as solid as every other (`AMOUNT`, by its even share of
its cell's room). Colour so gathers where energy is packed, moving or still: a still
tracer is a dot, a moving one a streak. A joint's marks count below 0: blue.

The turn is seen, read off the fluid: a mark is drawn more solid where its box's phase
stands near 0 and clearer half a turn away (`CREST`), so one bright crest a ring goes
round at the whirl's own speed. `ring` 0, 1 or 2 draws that ring's fluid alone. A mark
that would count under `FAINT` is too faint for the screen to add up: of such tracers the
right share is drawn at `FAINT` and the rest not, so the jet by the axis is still seen.
"""
import warp as wp

from physics.view.definitions.shape import drawn_place, even_share
from physics.view.logic.tracers import Tracers

AMOUNT = 0.05  # how solid one mark is drawn; only draws
CREST = 0.7    # how much the phase's crest stands out; only draws
FAINT = 0.3    # the least a drawn mark counts; only draws


@wp.kernel
def _spots(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
           placed: wp.array(dtype=wp.mat33), phase: wp.array2d(dtype=wp.float32),
           slot: wp.int32, ring: wp.int32, drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    mark = tail[(head + 1 + age) % tail.shape[0], tracer]
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    inside = wp.vec3(mark[0], mark[1] - float(cell * across), mark[2]) / float(across) - wp.vec3(0.5)
    spot = (drawn_place(inside, placed[cell]) + wp.vec3(0.5)) * float(across)
    box = (wp.clamp(int(mark[0]), 0, across - 1) * across + wp.clamp(
        int(mark[1]) - cell * across, 0, across - 1)) * across + wp.clamp(int(mark[2]), 0, across - 1)
    crest = 1.0 + CREST * wp.cos(phase[slot + cell, box])
    share = even_share(inside, placed[cell])
    if share < FAINT:
        chosen = float(tracer) * 0.6180339887
        share = FAINT * float(chosen - wp.floor(chosen) < share / FAINT)
    counts = wp.sign(placed[cell][2][2]) * AMOUNT * share * crest
    if ring >= 0 and int(mark[0]) / (across / 3) != ring:
        counts = 0.0
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], counts)


def line_streaks(tracers: Tracers, across: int, placed: wp.array, phase: wp.array,
                 slot: int = 0, ring: int = -1) -> wp.array:
    """(marks, tracers) of four floats: the place in the picture, how solid (below 0: blue).
    `phase` holds the cells' boxes' phases from slot `slot` on."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    wp.launch(_spots, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, phase, slot, ring, drawn])
    return drawn
