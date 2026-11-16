"""The engine: signed whole quanta and a fluid phase per Planck box, every rule neighbour-only, universes side by side, on the GPU.

Bible `physics/quantum-relativity`, rule 4: energy is the only countable thing. Each
Planck box holds one signed whole number (plus = pulling energy, an energy volume's box;
minus = pushing energy, a joint's box; 0 = no space) and the phase of the quantum fluid,
whose complex number is sqrt(|quanta|) e^{i phase}. Every rule reads only a box's
neighbours, so the engine scales to huge grids; the amount ignoring sign is conserved
exactly by integer transfers. One engine (issue #3): the core rules below, many
universes side by side in one GPU job (a batch stacked along the first axis, state.py,
the harness physics/bench measures it) and the detail ladder (physics/ladder) kept
beside the boxes and switched every step once attached (attach_ladder.py). The cells
form a forced cell lattice (owner, 2026-11-16): a cell's kind and neighbours are fixed
by its lattice index (an energy volume when all three are even, else a joint, so two
face-neighbouring volumes share exactly one joint), its corners move so its size
follows its energy, and each box belongs to the cell covering it, changing owner only
once its quanta flowed out. Quanta never change sign except in joint collapse.

Module map:
  settings.py       Settings: every rule's strength, all on by default
  state.py          State: quanta (int32), phase, pull, pull_rate, clock (float32) per box,
                    a batch of universes side by side, their seeds and start references
  uneven_field.py   uneven_field: a seeded Gaussian random field, ripples at every size
  new_universe.py   new_universe: the cell lattice of wrapped volumes, uneven whole quanta, margin
  lattice_kinds.py  lattice_kinds: every lattice cell's kind on the host
  cell_place.py     cell_place (Warp func): a cell's lattice indices from its id
  cell_kind.py      cell_kind (Warp func): volume, joint or ring from the lattice index
  shared_joint.py   shared_joint (Warp func): the one joint two face-neighbouring volumes share
  kuhn_corner.py    kuhn_corner (Warp func): which cell corner each tet corner is
  kuhn_tet.py       kuhn_tet (Warp func): a cell's six tetrahedra from its moving corners
  cell_score.py     cell_score (Warp func): how deep a point lies inside a moving cell
  face_step.py      face_step (Warp func): the six face steps
  inside.py         inside (Warp func): a box index in the reading box's own universe
  clock_rate.py     clock_rate (Warp func): a box's clock rate from its packing
  squeeze.py        squeeze (Warp func): a joint box's packing over its rest, minus 1
  openness.py       openness (Warp func): 1 between boxes of one sign, else 0 (hops)
  leak_openness.py  leak_openness (Warp func): a joint's openness to a leak, 0 sealed
  pass_seed.py      pass_seed: the rounding hash's seed per step, substep and pass
  pair_rotation.py  pair_rotation (Warp func): the exact two-box fluid step in whole quanta
  box_potential.py  box_potential (Warp func): energy per quantum (pull, own push, joint push)
  pull_wave.py      pull_wave: the pull's wave equation, leapfrog, light speed, plus-sourced
  on_site.py        on_site: exact phase turns by the box potential on local clocks
  hop.py            hop: exact pair rotations moving whole quanta inside one sign
  leak.py           leak: plus quanta across their shared joint, any thickness, one transfer
  substeps.py       substeps: fluid substeps per universe from its steepest same-sign face
  move_corners.py   move_corners: cell pressures and the pull move the lattice corners
  box_targets.py    box_targets: the cell whose moving region covers each box
  rehome_site.py    rehome_site (Warp func), rehome_dims: the 27 rehome phases' boxes
  one_group.py      one_group (Warp func): marked boxes round a box form one group
  block_adjacency.py block_adjacency: the 3 x 3 x 3 block's neighbour masks
  drain.py          drain: a leaving box empties into its own cell first
  hand_over.py      hand_over: the emptied box joins the cell covering it
  collapse.py       collapse: joint boxes past the limit turn plus, amount kept
  tick_clocks.py    tick_clocks: each box's own proper time
  totals.py         totals: exact plus, minus and total quanta, overall and per universe
  attach_ladder.py  attach_ladder: a detail ladder store beside the boxes
  ladder_sync.py    ladder_sync: write changed blocks, switch levels (once a step)
  step.py           step: pull wave, corners, fluid, rehoming, collapse, clocks, ladder
  host_boxes.py     host_boxes: one universe's arrays on the host
  lattice_of.py     lattice_of: one universe's cell lattice, to start a new State on it
  cells.py          cells: the lattice's energy volumes and joints, read off the box owners
  seams.py          seams: where two joints meet (boxes, faces, meeting pairs)
  stats.py          stats: amounts, drift, cells, lumps, seals, wrapping, web, edge, clocks
  view_source.py    view_source: occupied boxes for the viewer, compacted on the GPU
"""
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.uneven_field import uneven_field
from physics.engine.new_universe import new_universe
from physics.engine.lattice_kinds import lattice_kinds
from physics.engine.cell_place import cell_place
from physics.engine.cell_kind import cell_kind
from physics.engine.shared_joint import shared_joint
from physics.engine.kuhn_corner import kuhn_corner
from physics.engine.kuhn_tet import kuhn_tet
from physics.engine.cell_score import cell_score
from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.clock_rate import clock_rate
from physics.engine.squeeze import squeeze
from physics.engine.openness import openness
from physics.engine.leak_openness import leak_openness
from physics.engine.pass_seed import pass_seed
from physics.engine.pair_rotation import pair_rotation
from physics.engine.box_potential import box_potential
from physics.engine.pull_wave import pull_wave
from physics.engine.on_site import on_site
from physics.engine.hop import hop
from physics.engine.leak import leak
from physics.engine.substeps import substeps
from physics.engine.move_corners import move_corners
from physics.engine.box_targets import box_targets
from physics.engine.rehome_site import rehome_dims, rehome_site
from physics.engine.one_group import one_group
from physics.engine.block_adjacency import block_adjacency
from physics.engine.drain import drain
from physics.engine.hand_over import hand_over
from physics.engine.collapse import collapse
from physics.engine.tick_clocks import tick_clocks
from physics.engine.totals import totals
from physics.engine.attach_ladder import attach_ladder
from physics.engine.ladder_sync import ladder_sync
from physics.engine.step import step
from physics.engine.host_boxes import host_boxes
from physics.engine.lattice_of import lattice_of
from physics.engine.cells import ENERGY_VOLUME, JOINT, cells
from physics.engine.seams import seams
from physics.engine.stats import stats
from physics.engine.view_source import view_source

__all__ = ["Settings", "State", "uneven_field", "new_universe", "lattice_kinds", "cell_place", "cell_kind",
           "shared_joint", "kuhn_corner", "kuhn_tet", "cell_score", "face_step", "inside", "clock_rate", "squeeze", "openness",
           "leak_openness", "pass_seed", "pair_rotation", "box_potential", "pull_wave", "on_site", "hop", "leak", "substeps",
           "move_corners", "box_targets", "rehome_dims", "rehome_site", "one_group", "block_adjacency", "drain", "hand_over", "collapse", "tick_clocks", "totals",
           "attach_ladder", "ladder_sync", "step", "host_boxes", "lattice_of", "ENERGY_VOLUME", "JOINT", "cells", "seams", "stats",
           "view_source"]
