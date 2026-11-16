"""The engine: signed whole quanta and a fluid phase per Planck box, every rule neighbour-only, universes side by side, on the GPU.

Bible `physics/quantum-relativity`, rule 4: energy is the only countable thing. Each
Planck box holds one signed whole number (plus = pulling energy, an energy volume's box;
minus = pushing energy, a joint's box; 0 = no space) and the phase of the quantum fluid,
whose complex number is sqrt(|quanta|) e^{i phase}. Every rule reads only a box's
neighbours, so the engine scales to huge grids; the amount ignoring sign is conserved
exactly by integer transfers. One engine (issue #3): the core rules below, many
universes side by side in one GPU job (a batch stacked along the first axis, state.py,
the harness physics/bench measures it) and the detail ladder (physics/ladder) kept
beside the boxes and switched every step once attached (attach_ladder.py). Cells are
not labels the engine keeps: energy volumes are connected plus regions and joints the
minus web, read off the signs on the host (cells.py).

Module map:
  settings.py       Settings: every rule's strength, all on by default
  state.py          State: quanta (int32), phase, pull, pull_rate, clock (float32) per box,
                    a batch of universes side by side, their seeds and start references
  uneven_field.py   uneven_field: a seeded Gaussian random field, ripples at every size
  new_universe.py   new_universe: wrapped volumes in a joint web, uneven whole quanta, margin
  brick_start.py    brick_start: wrapped blocks of plus boxes in a minus web, empty margin
  face_step.py      face_step (Warp func): the six face steps
  inside.py         inside (Warp func): a box index in the reading box's own universe
  clock_rate.py     clock_rate (Warp func): a box's clock rate from its packing
  squeeze.py        squeeze (Warp func): a joint box's packing over its rest, minus 1
  openness.py       openness (Warp func): 1 inside a kind, graded by squeeze across, 0 sealed
  box_potential.py  box_potential (Warp func): energy per quantum (pull, own push, joint push)
  box_energy.py     box_energy (Warp func): its integral, the energy claims lower
  pull_wave.py      pull_wave: the pull's wave equation, leapfrog, light speed, plus-sourced
  on_site.py        on_site: exact phase turns by the box potential on local clocks
  hop.py            hop: exact pair rotations moving whole quanta; friction, drift spreading
  claim.py          claim: border boxes change kind by whole-quanta moves, wrapping kept
  collapse.py       collapse: joint boxes past the limit turn plus, amount kept
  tick_clocks.py    tick_clocks: each box's own proper time
  totals.py         totals: exact plus, minus and total quanta, overall and per universe
  attach_ladder.py  attach_ladder: a detail ladder store beside the boxes
  ladder_sync.py    ladder_sync: write changed blocks, switch levels (once a step)
  step.py           step: pull wave, fluid, claims, collapse, clocks, ladder
  host_boxes.py     host_boxes: one universe's arrays on the host
  cells.py          cells: energy volumes (26-connected plus) and joints (minus) per box
  stats.py          stats: amounts, drift, cells, lumps, seals, wrapping, web, edge, clocks
  view_source.py    view_source: occupied boxes for the viewer, compacted on the GPU
"""
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.uneven_field import uneven_field
from physics.engine.new_universe import new_universe
from physics.engine.brick_start import brick_start
from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.clock_rate import clock_rate
from physics.engine.squeeze import squeeze
from physics.engine.openness import openness
from physics.engine.box_potential import box_potential
from physics.engine.box_energy import box_energy
from physics.engine.pull_wave import pull_wave
from physics.engine.on_site import on_site
from physics.engine.hop import hop
from physics.engine.claim import claim
from physics.engine.collapse import collapse
from physics.engine.tick_clocks import tick_clocks
from physics.engine.totals import totals
from physics.engine.attach_ladder import attach_ladder
from physics.engine.ladder_sync import ladder_sync
from physics.engine.step import step
from physics.engine.host_boxes import host_boxes
from physics.engine.cells import ENERGY_VOLUME, JOINT, cells
from physics.engine.stats import stats
from physics.engine.view_source import view_source

__all__ = ["Settings", "State", "uneven_field", "new_universe", "brick_start", "face_step", "inside", "clock_rate", "squeeze", "openness",
           "box_potential", "box_energy", "pull_wave", "on_site", "hop", "claim", "collapse", "tick_clocks", "totals",
           "attach_ladder", "ladder_sync", "step", "host_boxes", "ENERGY_VOLUME", "JOINT", "cells", "stats",
           "view_source"]
