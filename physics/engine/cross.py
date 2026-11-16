"""Cross: energy crosses from energy volume to energy volume through the one joint they share, bent by the pull; the crossing is the only friction and makes heat.

Contract: `cross(state, settings, squeeze, pull)` moves plus quanta across every joint
that lies between two energy volumes (state.between). Energy is an always-moving fluid
and more packed pulls harder (axioms 1 and 5; owner, 2026-11-16: the crossing is steered
by the pull), so the crossing is drift and diffusion together, the Smoluchowski law
J = -D (grad n + n grad(pull) / T), in its exact two-cell (Scharfetter-Gummel) form: in
one step the joint carries
  flow = crossing_rate * openness * (B(x) * Q0 - B(-x) * Q1) * step_time * rate
quanta from between[0] to between[1], Q0 and Q1 the two volumes' quanta, B(x) =
x / (e**x - 1) (B(0) = 1: equal pulls give plain diffusion), x = (pull1 - pull0) / T
with pull0, pull1 the volumes' mean pull per quantum (`pull`, cell_count.py; lower is
deeper, nearer the packed side) and T the energy of the fluid's own motion per quantum
along one axis, pooled over both volumes (jiggle.py; Einstein's relation, mobility =
D / T): the moving energy is bent toward the volume that pulls harder, so energy
attracts energy, and its own heat spreads it. No new setting. `rate` is the joint's own
clock (state.rate); openness is 1 - squeeze / seal_squeeze, between 0 and 1 (`squeeze`
per cell from cell_laws.py: the more squeezed the joint, the more it resists, and past
the seal nothing passes, so packed lumps seal themselves). A joint never carries more
than its giving volume holds. The flow is spread over the joint's crossings: a crossing
is a straight line along an axis from a plus box of one of its volumes through one or
more minus boxes, all of that joint, to a plus box of the other (any thickness; round
the edge when the universe wraps). Each crossing hands floor(|flow| / crossings +
uniform) whole quanta from the end box on the giving side to the end box on the other
side, at most all but 1 of the giver's quanta: the quanta pass through the joint in one
handover and never take the joint's sign, so plus and minus never cancel (bible rule 4).
Friction: the arriving quanta come without their direction, at a random phase (the hash
of the box, the universe's key and the step), so the box's phase becomes
arg(sqrt(n) e^{i phase} + sqrt(m) e^{i random}) for n quanta held and m arrived: directed
flow turned to jiggling, heat, and no quantum lost. Phases inside cells are not touched.
state.crossed gets each joint's net quanta from between[0] to between[1] this step and
tallies[u, 0] every quantum that crossed. Joints facing empty space or a wall carry
nothing. Returns None.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.jiggle import jiggle
from physics.engine.pass_seed import pass_seed
from physics.engine.settings import Settings
from physics.engine.state import State

PI = 3.14159265358979
_COUNT, _MOVE = 0, 1
_EXP_LIMIT = 700.0  # numerics: e**700 is near float64's top; beyond it B is 0 or |x| to the last digit


@wp.func
def _place(home: int, side: int, axis: int, first: int, second: int, along: int) -> wp.vec3i:
    """The box at `along` (taken round the side) on the line along `axis` through (first, second)."""
    spot = along % side
    if axis == 0:
        return wp.vec3i(home * side + spot, first, second)
    if axis == 1:
        return wp.vec3i(home * side + first, spot, second)
    return wp.vec3i(home * side + first, second, spot)


@wp.func
def _hand(quanta: wp.array3d(dtype=wp.int32), arrived: wp.array3d(dtype=wp.int32), giver: wp.vec3i, taker: wp.vec3i,
          moved: int) -> int:
    """Hand up to `moved` plus quanta from giver to taker, leaving the giver at least 1; returns what moved.

    Only this line's thread touches its boxes in one axis pass, so the read is not racing.
    """
    moved = wp.min(moved, quanta[giver[0], giver[1], giver[2]] - 1)
    if moved <= 0:
        return 0
    wp.atomic_add(quanta, giver[0], giver[1], giver[2], -moved)
    wp.atomic_add(quanta, taker[0], taker[1], taker[2], moved)
    wp.atomic_add(arrived, taker[0], taker[1], taker[2], moved)
    return moved


@wp.kernel
def _lines(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), kinds: wp.array2d(dtype=wp.int32),
           between: wp.array2d(dtype=wp.vec2i), flow: wp.array2d(dtype=wp.float32), side: int, wrap: int, axis: int,
           mode: int, seed: int, keys: wp.array(dtype=wp.int32), lines: wp.array2d(dtype=wp.int32),
           crossed: wp.array2d(dtype=wp.int64), tallies: wp.array2d(dtype=wp.int64),
           arrived: wp.array3d(dtype=wp.int32)):
    home, first, second = wp.tid()
    length = side
    if wrap == 1:
        length = 2 * side
    last_plus = int(-1)
    joint_boxes = int(0)
    joint_cell = int(-1)
    for along in range(length):
        place = _place(home, side, axis, first, second, along)
        held = quanta[place[0], place[1], place[2]]
        cell = owner[place[0], place[1], place[2]]
        if held == 0:
            last_plus = -1
            joint_boxes = 0
        elif held < 0:
            if last_plus >= 0:
                if joint_boxes == 0:
                    joint_cell = cell
                elif joint_cell != cell:
                    joint_cell = -2
                joint_boxes += 1
        else:
            if last_plus >= 0 and joint_boxes > 0 and joint_cell >= 0 and last_plus < side:
                near = _place(home, side, axis, first, second, last_plus)
                low = owner[near[0], near[1], near[2]]
                sides = between[home, joint_cell]
                if kinds[home, joint_cell] == -1 and low != cell and sides[0] >= 0 and (
                        (sides[0] == low and sides[1] == cell) or (sides[0] == cell and sides[1] == low)):
                    if mode == 0:
                        wp.atomic_add(lines, home, joint_cell, 1)
                    else:
                        share = flow[home, joint_cell] / float(wp.max(lines[home, joint_cell], 1))
                        index = ((near[0] - home * side) * side + near[1]) * side + near[2]
                        rng = wp.rand_init(seed ^ keys[home], index * 3 + axis)
                        moved = int(wp.floor(wp.abs(share) + wp.randf(rng)))
                        giver = near
                        taker = place
                        if (share > 0.0) != (low == sides[0]):
                            giver = place
                            taker = near
                        moved = _hand(quanta, arrived, giver, taker, moved)
                        if moved > 0:
                            signed = moved
                            if share < 0.0:
                                signed = -moved
                            wp.atomic_add(crossed, home, joint_cell, wp.int64(signed))
                            wp.atomic_add(tallies, home, 0, wp.int64(moved))
            last_plus = along
            joint_boxes = 0


@wp.kernel
def _heat(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), arrived: wp.array3d(dtype=wp.int32),
          side: int, seed: int, keys: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    came = arrived[ti, tj, tk]
    if came <= 0:
        return
    home = ti // side
    held = wp.abs(quanta[ti, tj, tk])
    came = wp.min(came, held)  # a box that took and then gave on another line keeps what it holds
    kept = held - came
    index = ((ti - home * side) * side + tj) * side + tk
    rng = wp.rand_init(seed ^ keys[home], index)
    angle = 2.0 * PI * wp.randf(rng)
    own = phase[ti, tj, tk]
    real = wp.sqrt(float(kept)) * wp.cos(own) + wp.sqrt(float(came)) * wp.cos(angle)
    imaginary = wp.sqrt(float(kept)) * wp.sin(own) + wp.sqrt(float(came)) * wp.sin(angle)
    phase[ti, tj, tk] = wp.atan2(imaginary, real)


def cross(state: State, settings: Settings, squeeze: np.ndarray, pull: np.ndarray) -> None:
    """Move plus quanta across every joint between two volumes, drift down the pull and diffusion; arrivals become heat."""
    device = state.quanta.device
    state.crossed = np.zeros(state.kind.shape, dtype=np.int64)
    flow = _flow(state, settings, squeeze, pull)
    if not np.any(flow):
        return
    shape = state.kind.shape
    lines = wp.zeros(shape, dtype=wp.int32, device=device)
    crossed = wp.zeros(shape, dtype=wp.int64, device=device)
    arrived = wp.zeros(state.shape, dtype=wp.int32, device=device)
    flows = wp.array(flow.astype(np.float32), dtype=wp.float32, device=device)
    seed = pass_seed(state.step, 0, 40)
    for mode in (_COUNT, _MOVE):
        for axis in range(3):
            wp.launch(_lines, dim=(state.universes, state.side, state.side), device=device,
                      inputs=[state.quanta, state.owner, state.cells_d["kind"], state.cells_d["between"], flows,
                              state.side, int(state.wrap), axis, mode, int(seed) + axis, state.keys, lines, crossed,
                              state.tallies, arrived])
    wp.launch(_heat, dim=state.shape, device=device,
              inputs=[state.quanta, state.phase, arrived, state.side, int(pass_seed(state.step, 0, 41)), state.keys])
    state.crossed = crossed.numpy()


def _flow(state: State, settings: Settings, squeeze: np.ndarray, pull: np.ndarray) -> np.ndarray:
    """Quanta each joint carries this step from between[0] to between[1] (float, signed)."""
    rows, columns = np.nonzero((state.kind == -1) & (state.between[..., 1] >= 0) & (state.between[..., 0] >= 0))
    flow = np.zeros(state.kind.shape)
    if rows.size == 0:
        return flow
    sides = state.between[rows, columns]
    amounts = state.cell_quanta.astype(np.float64)
    giving, taking = amounts[rows, sides[:, 0]], amounts[rows, sides[:, 1]]
    lean = _lean(state, pull, rows, sides)
    openness = np.clip(1.0 - np.maximum(squeeze[rows, columns], 0.0) / settings.seal_squeeze, 0.0, 1.0)
    carried = (settings.crossing_rate * openness * (_bernoulli(lean) * giving - _bernoulli(-lean) * taking)
               * settings.step_time * state.rate[rows, columns])
    flow[rows, columns] = np.clip(carried, -taking, giving)
    return flow


def _lean(state: State, pull: np.ndarray, rows: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """x = (pull of between[1] - pull of between[0]) / T, T the two volumes' pooled jiggle; inf-safe."""
    spread, weight = jiggle(state)
    pooled = spread[rows, sides[:, 0]] + spread[rows, sides[:, 1]]
    weights = weight[rows, sides[:, 0]] + weight[rows, sides[:, 1]]
    heat = np.where(weights > 0.0, pooled / np.maximum(weights, 1e-300), 0.0)
    rise = pull[rows, sides[:, 1]] - pull[rows, sides[:, 0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        lean = np.where(rise == 0.0, 0.0, rise / heat)
    return np.clip(lean, -_EXP_LIMIT, _EXP_LIMIT)


def _bernoulli(lean: np.ndarray) -> np.ndarray:
    """B(x) = x / (e**x - 1), B(0) = 1, without overflow."""
    small = np.abs(lean) < 1e-8
    safe = np.where(small, 1.0, lean)
    with np.errstate(over="ignore"):
        value = safe / np.expm1(safe)
    return np.where(small, 1.0 - 0.5 * lean, value)
