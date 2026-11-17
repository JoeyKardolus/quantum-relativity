"""The engine: a perfect grid of cells that have only relations and a size, each with its own block of compute boxes, on the GPU.

Bible `physics/quantum-relativity`, the eight axioms. The cells are the grid (axiom 3,
owner 2026-11-17): a perfect A B A B pattern, an energy volume where all three lattice
indices are even and a joint everywhere else, so every volume is wrapped by joints on
all 26 sides and the joints are one web. A cell has no geometry: only its neighbours
and a length set by its energy alone (constant / quanta, more energy smaller). Planck
boxes are not space: each cell owns a fixed block of 216 compute boxes holding whole
signed quanta and a phase, only to work out its fluid and whirls; no box ever changes
cell. Energy is whole quanta, plus pulls and minus pushes; the total is not fixed: a
squeezed joint makes minus quanta out of itself, a collapse takes the joint's energy
off the rest, and the books are exact.

Module map:
  settings.py       Settings: every rule's strength, all on by default
  state.py          State: per cell its block of boxes, quanta, length, clock, pull; the
                    relations (pristine cells from the lattice, changed cells listed)
  lattice_step.py, neighbour.py, way.py  Warp funcs: a cell's neighbours and a joint's ways
  uneven_field.py   uneven_field: a seeded Gaussian random field, ripples at every size
  new_universe.py   new_universe: the perfect grid with slightly uneven energy
  cell_laws.py      cell_laws: size, packing, clock rate from quanta; volumes to tear
  clock_rate.py, tick_clocks.py  the local clock's rate and its ticking
  pull_wave.py      pull_wave: the pull spreads cell to cell at light speed
  jiggle.py         jiggle: an energy volume's own heat
  squeeze.py        squeeze: pull across a joint against its push; squeezed joints make energy
  cross.py          cross: plus quanta cross through the joint between two volumes
  pour.py           pour: what a cell gave and took leaves and enters its own boxes
  on_site.py, site_turn.py, fluid_potential.py, hop.py, hop_pair.py, pair_rotation.py,
  pass_seed.py      the frictionless fluid inside each cell's block
  events.py, collapse.py, tear.py  the only changes of who neighbours whom
  step.py           step: one step, every rule on its cell's clock
  quantum*.py       axiom 7: moving energies tracked whole, splits, pointer, settling
  totals.py         totals: the exact books from the boxes
  pattern.py        pattern: counts every break of the A B A B pattern
  way_list.py       way_list: every way now, as arrays
  cells.py          cells: per-cell host arrays for scenes and the viewer
  stats.py          stats: cells, books, events, the universe's size, lumps
"""
from physics.engine.cell_laws import cell_laws
from physics.engine.cells import cells
from physics.engine.new_universe import new_universe
from physics.engine.pattern import pattern
from physics.engine.quantum import quantum
from physics.engine.quantum_hand import quantum_hand
from physics.engine.quantum_move import quantum_move
from physics.engine.quantum_packet import quantum_packet
from physics.engine.quantum_react import quantum_react
from physics.engine.quantum_settle import quantum_settle
from physics.engine.quantum_store import quantum_store
from physics.engine.settings import Settings
from physics.engine.state import BLOCK, BOXES, ENERGY_VOLUME, JOINT, State
from physics.engine.stats import stats
from physics.engine.step import step
from physics.engine.totals import totals
from physics.engine.uneven_field import uneven_field
from physics.engine.way_list import way_list

__all__ = ["BLOCK", "BOXES", "ENERGY_VOLUME", "JOINT", "Settings", "State", "cell_laws", "cells", "new_universe",
           "pattern", "quantum", "quantum_hand", "quantum_move", "quantum_packet", "quantum_react",
           "quantum_settle", "quantum_store", "stats", "step", "totals", "uneven_field", "way_list"]
