"""The engine: signed whole quanta and a fluid phase per Planck box on a graph of cells, every box rule neighbour-only, universes side by side, on the GPU.

Bible `physics/quantum-relativity`, rule 4: energy is the only countable thing. Each
Planck box holds one signed whole number (plus = an energy volume's box, minus = a
joint's box, 0 = no space) and the phase of the quantum fluid, whose complex number is
sqrt(|quanta|) e^{i phase}. The amount ignoring sign is conserved exactly by integer
transfers. The cells are a graph (owner, 2026-11-16): each box belongs to one cell, each
cell keeps its own neighbours (a joint its two energy volumes, a volume its joints), so
one volume can split, or two volumes become one, without touching any other cell; the
boxes keep ABAB (every volume wrapped by joints on all 26 sides, two volumes never
touching, the joints thin mortar in one web). A cell's size follows its energy (length =
constant / energy, the constant fixed once by the start's correction); a volume only
pulls, a joint only pushes and swings like a spring, empty space does nothing; energy
crosses from volume to volume through the joint they share, drifting down the pull
and diffusing by its own jiggle, and the crossing is the
only friction, making heat; everything in a cell runs on its local clock. Quanta change
sign only in joint collapse (minus to plus) and in a volume split's tear (plus to
minus). The detail ladder (physics/ladder) can be kept beside the boxes (attach_ladder.py).

Module map:
  settings.py       Settings: every rule's strength, all on by default
  state.py          State: per box quanta (int32), phase, pull, pull_rate, clock, owner,
                    marks; the cell graph per universe; a batch of universes side by side
  uneven_field.py   uneven_field: a seeded Gaussian random field, ripples at every size
  new_universe.py   new_universe: wrapped volumes in a thin joint web, sizes corrected to energy
  reach.py          reach (Warp func): a neighbour box in the same universe, round wrapping edges
  face_step.py      face_step (Warp func): the six face steps
  near.py           near: boxes with a marked neighbour, on the host
  clock_rate.py     clock_rate (Warp func): a clock's rate from its packing
  pass_seed.py      pass_seed: the rounding hash's seed per step, substep and pass
  pair_rotation.py  pair_rotation (Warp func): the exact two-box fluid step in whole quanta
  box_potential.py  box_potential (Warp func): energy per quantum (pull for plus, push for minus)
  fluid_potential.py fluid_potential (Warp func): box_potential plus the fluid's own pressure in its cell
  cell_count.py     cell_count: every cell's quanta, boxes and mean pull
  cell_laws.py      cell_laws: clock rate, wanted size, volume size, joint push and squeeze
  cell_upload.py    cell_upload: the cell table on the GPU for the box kernels
  cell_room.py      cell_room: (Warp func) the share of a cell's size its boxes lack
  settle_counts.py  settle_counts: a phase's box-count changes applied at once, order-free
  swing_table.py    swing_table: each volume's swinging joints in one fixed order, kept on the GPU
  joint_swing.py    joint_swing: joints' sizes swing like frictionless springs, energy kept
  speed_handover.py speed_handover: a joint that stops swinging hands its speed to its spring neighbours
  pull_wave.py      pull_wave: the pull's wave equation, leapfrog, light speed, local clocks
  on_site.py        on_site: exact phase turns by the box potential on local clocks
  hop.py            hop: exact pair rotations moving whole quanta inside one cell
  site_turn.py      site_turn (Warp func): one box's phase turned by its potential
  hop_pair.py       hop_pair (Warp func): one exact pair rotation between two boxes of a cell
  substeps.py       substeps: fluid substeps per cell from its own steepest face
  later_substeps.py later_substeps: the substeps after the first, over the listed boxes only
  jiggle.py         jiggle: the fluid's own motion per quantum in each volume
  cross.py          cross: drift down the pull and diffusion through a joint, heat
  quantum_store.py  quantum_store: the moving energies tracked whole and how they ended (axiom 7)
  quantum_packet.py quantum_packet: name part of a volume's energy as one moving energy
  quantum_joints.py quantum_joints: a volume's joints, kept to notice a collapse or tear
  quantum_move.py   quantum_move: whole quanta from one volume's boxes to another's, any distance
  quantum_hand.py   quantum_hand: frictionless handover between neighbours (splitter, merge back; staged)
  quantum_settle.py quantum_settle: a moving energy settles whole in its pointer's cell
  quantum_react.py  quantum_react: a placed reaction settles what has a part in a cell
  quantum.py        quantum: the step's part: waves and oscillations turn, pointers jump, settles
  exchange.py       exchange: wall boxes follow the cells' sizes, ABAB kept, light speed
  mortar_shift.py   mortar_shift: a growing volume pushes a thin joint wall into its neighbour
  group_label.py    group_label: connected groups of boxes, joined round wrapping edges
  one_group.py      one_group (Warp func): marked boxes round a box form one group
  block_adjacency.py block_adjacency: the 3 x 3 x 3 block's neighbour masks
  collapse.py       collapse: a joint past the limit gives way, its volumes become one
  cells_span.py     cells_span: the room a few cells' boxes lie in, measured on the GPU
  window_read.py    window_read: one room's quanta, owners and marks on the host
  window_write.py   window_write: such a room written back
  volume_split.py   volume_split: a too-big volume tears in two, only it
  wall_press.py     wall_press: the volumes' fluid pressure on each joint's faces
  tick_clocks.py    tick_clocks: each box's and cell's own proper time
  totals.py         totals: exact plus, minus and total quanta, overall and per universe
  attach_ladder.py  attach_ladder: a detail ladder store beside the boxes
  ladder_sync.py    ladder_sync: write changed blocks, switch levels (once a step)
  step.py           step: cell laws, collapse, split, pull, fluid, crossing, swing, exchange, clocks
  host_boxes.py     host_boxes: one universe's arrays and cell graph on the host
  lattice_of.py     lattice_of: one universe's cell graph, to start a new State on it
  cells.py          cells: the energy volumes and joints, read off the boxes
  seams.py          seams: where two joints meet (boxes, faces, meeting pairs)
  stats.py          stats: amounts, drift, cells, lumps, seals, wrapping, web, edge, clocks
  view_source.py    view_source: occupied boxes for the viewer, compacted on the GPU
"""
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.uneven_field import uneven_field
from physics.engine.new_universe import new_universe
from physics.engine.reach import reach
from physics.engine.face_step import face_step
from physics.engine.near import near
from physics.engine.clock_rate import clock_rate
from physics.engine.pass_seed import pass_seed
from physics.engine.pair_rotation import pair_rotation
from physics.engine.box_potential import box_potential
from physics.engine.fluid_potential import fluid_potential
from physics.engine.cell_count import cell_count
from physics.engine.cell_laws import cell_laws
from physics.engine.cell_upload import cell_upload
from physics.engine.cell_room import cell_room
from physics.engine.settle_counts import settle_counts
from physics.engine.swing_table import swing_table
from physics.engine.speed_handover import speed_handover
from physics.engine.joint_swing import joint_swing
from physics.engine.pull_wave import pull_wave
from physics.engine.site_turn import site_turn
from physics.engine.hop_pair import hop_pair
from physics.engine.on_site import on_site
from physics.engine.hop import hop
from physics.engine.substeps import substeps
from physics.engine.later_substeps import later_substeps
from physics.engine.cross import cross
from physics.engine.jiggle import jiggle
from physics.engine.quantum_store import quantum_store
from physics.engine.quantum_packet import quantum_packet
from physics.engine.quantum_joints import quantum_joints
from physics.engine.quantum_move import quantum_move
from physics.engine.quantum_hand import quantum_hand
from physics.engine.quantum_settle import quantum_settle
from physics.engine.quantum_react import quantum_react
from physics.engine.quantum import quantum
from physics.engine.exchange import exchange
from physics.engine.mortar_shift import mortar_shift
from physics.engine.group_label import group_label
from physics.engine.one_group import one_group
from physics.engine.block_adjacency import block_adjacency
from physics.engine.collapse import collapse
from physics.engine.cells_span import cells_span
from physics.engine.window_read import window_read
from physics.engine.window_write import window_write
from physics.engine.volume_split import volume_split
from physics.engine.wall_press import wall_press
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

__all__ = ["Settings", "State", "uneven_field", "new_universe", "reach", "face_step", "near", "clock_rate",
           "pass_seed", "pair_rotation", "box_potential", "fluid_potential", "cell_count", "cell_laws", "cell_upload", "cell_room", "settle_counts", "swing_table", "speed_handover", "joint_swing",
           "pull_wave", "site_turn", "hop_pair", "on_site", "hop", "substeps", "later_substeps", "jiggle", "cross", "quantum_store", "quantum_packet", "quantum_joints", "quantum_move", "quantum_hand",
           "quantum_settle", "quantum_react", "quantum", "exchange", "mortar_shift", "group_label", "one_group", "block_adjacency",
           "collapse", "cells_span", "window_read", "window_write", "volume_split", "wall_press", "tick_clocks", "totals", "attach_ladder", "ladder_sync", "step",
           "host_boxes", "lattice_of", "ENERGY_VOLUME", "JOINT", "cells", "seams", "stats", "view_source"]
