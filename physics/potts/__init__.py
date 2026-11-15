"""The cell engine: the foam's cells as groups of fixed Planck boxes on the GPU (Warp), the cellular Potts method.

Bible `physics/quantum-relativity`, "How the engine holds the foam": a fixed grid of
Planck boxes carries the energy as a fluid; energy volumes (A) and joints (B) are
groups of boxes in a 3D checkerboard; a cell's size is how many boxes it holds, and it
grows or shrinks by claiming or giving up boxes at its border, so its size follows its
energy (Graner and Glazier, 1992). Named `potts`, not `cells`: an earlier
`physics/cells` engine lives in git history.

Module map:
  settings.py         Settings: every rule's strength, all_on() defaults
  universe.py         Universe: box owner, energy (exact quanta), momentum; cell kind, sums, wants, clocks; contacts
  new_universe.py     new_universe: the checkerboard of k**3-box cells, uneven energy, in an empty margin
  contact_pairs.py    contact_pairs: which cells touch across how many box faces (host)
  contact_table.py    contact_table: those pairs laid out per cell for the GPU
  box_owner.py        box_owner (Warp func): a box's cell, -1 empty or outside
  direction.py        direction (Warp func): the six face steps
  border_openness.py  border_openness (Warp func): 1 inside a cell, through a joint down to 0 at the seal
  cell_sums.py        cell_sums: energy, size, potential, want, compression, clock per cell; the A source
  uneven_field.py     uneven_field: seeded ripples at every size (FFT, numpy), the start's unevenness
  multigrid.py        Multigrid: the GPU Poisson solver, open-space boundary
  pull.py             pull: the A-energy's potential (multigrid V-cycles, CUDA graph)
  fastest_flow.py     fastest_flow: the substep's speed bound
  kick.py             kick: the pull on A, the fluid's pressure inside a cell
  flow.py             flow: exact donor-cell transport; friction and the seal at joint borders
  box_moves.py        box_moves: eight parity phases of border moves (size, pull, push; ABAB, contacts, connected)
  joint_collapse.py   joint_collapse: a joint past its limit vanishes, its energy volumes fall together
  step.py             step: sums, pull, fluid substeps, box moves, collapses
  stats.py            stats: energy and drift, cells, joints, collapses, occupied growth, lumps, seals, ABAB
  view_source.py      view_source: per occupied box its place, energy, kind, border flag (GPU)
"""
from physics.potts.settings import Settings
from physics.potts.universe import ENERGY_VOLUME, JOINT, QUANTUM, SLOTS, Universe
from physics.potts.uneven_field import uneven_field
from physics.potts.multigrid import Multigrid
from physics.potts.new_universe import new_universe
from physics.potts.contact_pairs import contact_pairs
from physics.potts.contact_table import contact_table
from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.border_openness import border_openness
from physics.potts.cell_sums import cell_sums
from physics.potts.pull import pull
from physics.potts.fastest_flow import fastest_flow
from physics.potts.kick import kick
from physics.potts.flow import flow
from physics.potts.box_moves import box_moves
from physics.potts.joint_collapse import joint_collapse
from physics.potts.step import step
from physics.potts.stats import stats
from physics.potts.view_source import view_source

__all__ = ["Settings", "ENERGY_VOLUME", "JOINT", "QUANTUM", "SLOTS", "uneven_field", "Multigrid", "Universe", "new_universe", "contact_pairs",
           "contact_table", "box_owner", "direction", "border_openness", "cell_sums", "pull", "fastest_flow", "kick",
           "flow", "box_moves", "joint_collapse", "step", "stats", "view_source"]
