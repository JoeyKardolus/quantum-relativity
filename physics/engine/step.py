"""Step: one step of the whole-quanta engine on its graph of cells.

Contract: `step(state, settings)` advances the state by one step_time of the counter;
every part runs on the local clock of the cell it acts in (owner, 2026-11-16):
  1. cell_count, cell_laws  every cell's quanta and boxes are counted, its clock rate,
                   wanted size (length = constant / energy), a volume's size and a
                   joint's push and squeeze set (cell_count.py, cell_laws.py)
  2. collapse      joints squeezed past the limit collapse, their two volumes one cell
                   (collapse.py); volume_split: the biggest volume past split_across in
                   each universe tears in two (volume_split.py); the cells are counted
                   again when either happened
  3. pull_wave     the pull moves one leapfrog step at light speed (pull_wave.py)
  4. fluid         cut into equal substeps per universe, as many as its steepest face
                   inside a cell needs (substeps.py; numerics), each a Strang splitting:
                   on_site for half a step, the hops along x, y, z (parity 0 then 1) for
                   half a step each, the hops in reverse order, on_site for half a step
                   (on_site.py, hop.py: whole quanta inside one cell, frictionless)
  5. cross         energy diffuses across every joint between two volumes, the arriving
                   quanta turned to heat (cross.py)
  6. joint_swing   every joint's size swings, pushed against its neighbours'
                   (joint_swing.py)
  7. exchange      the boxes follow the cells' sizes, at most at light speed
                   (exchange.py); mortar_shift: a growing volume pushes a one-box-thick
                   joint wall on into its neighbour (mortar_shift.py)
  8. tick_clocks   every box's and cell's clock advances (tick_clocks.py)
then counts the step and, when a detail ladder is attached (attach_ladder.py), writes the
changed blocks into it and switches every block's level (ladder_sync.py). The amount
ignoring sign is exact through every part, and quanta keep their sign except in
collapse and a tear. `marks` is zeroed first; after a step it says which boxes changed
cell or sign and by which rule (1 collapse, 2 owner change, 8 split). Returns None.
"""
from __future__ import annotations

from physics.engine.cell_count import cell_count
from physics.engine.cell_laws import cell_laws
from physics.engine.cell_upload import cell_upload
from physics.engine.collapse import collapse
from physics.engine.cross import cross
from physics.engine.exchange import exchange
from physics.engine.hop import hop
from physics.engine.joint_swing import joint_swing
from physics.engine.ladder_sync import ladder_sync
from physics.engine.mortar_shift import mortar_shift
from physics.engine.on_site import on_site
from physics.engine.pull_wave import pull_wave
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.substeps import substeps
from physics.engine.tick_clocks import tick_clocks
from physics.engine.volume_split import volume_split

_ORDER = [(axis, parity) for axis in range(3) for parity in range(2)]


def step(state: State, settings: Settings) -> None:
    """One step: cell laws, collapse and split, pull wave, fluid, crossing, joint swing, boxes follow sizes, clocks."""
    state.marks.zero_()
    cell_count(state)
    squeeze = cell_laws(state, settings)
    if collapse(state, settings, squeeze) + volume_split(state, settings):
        cell_count(state)
        squeeze = cell_laws(state, settings)
    cell_upload(state)
    pull_wave(state, settings)
    half = 0.5 * settings.step_time
    for sub_index in range(int(substeps(state, settings).max())):
        _fluid(state, settings, half, sub_index)
    cross(state, settings, squeeze)
    joint_swing(state, settings, cell_count(state))
    cell_upload(state)
    exchange(state, settings)
    mortar_shift(state, settings)
    tick_clocks(state, settings)
    state.step += 1
    ladder_sync(state)


def _fluid(state: State, settings: Settings, half: float, sub_index: int) -> None:
    """One fluid substep, Strang split; each kernel divides `half` by its universe's substep count."""
    on_site(state, settings, half, sub_index)
    for pass_index, (axis, parity) in enumerate(_ORDER):
        hop(state, settings, axis, parity, half, pass_index, sub_index)
    for pass_index, (axis, parity) in enumerate(_ORDER[::-1]):
        hop(state, settings, axis, parity, half, 6 + pass_index, sub_index)
    on_site(state, settings, half, sub_index)
