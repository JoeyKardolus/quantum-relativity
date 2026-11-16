"""The cell engine: the foam's cells as groups of fixed Planck boxes on the GPU (Warp), the cellular Potts method.

Bible `physics/quantum-relativity`, "How the engine holds the foam": a fixed grid of
Planck boxes carries the energy as a quantum fluid, one complex number per box
(Gross-Pitaevskii); energy volumes (A) and joints (B) are groups of boxes, every
energy volume fully wrapped in joints and the joints one web; a cell's size is how many boxes it holds, and it
grows or shrinks by claiming or giving up boxes at its border, so its size follows its
energy (Graner and Glazier, 1992). Named `potts`, not `cells`: an earlier
`physics/cells` engine lives in git history.

Module map:
  settings.py         Settings: every rule's strength, all_on() defaults
  universe.py         Universe: boxes (owner, psi, heat), cells (kind, sums, wants, clocks, swing)
  new_universe.py     new_universe: k**3-box (k 10) energy volumes wrapped in a web of k**3-box joints, uneven
  contact_pairs.py    contact_pairs: which cells touch across how many box faces (host)
  touch_pairs.py      touch_pairs: which cells touch at a face, edge or corner, empty space as -1 (host)
  region_host.py      region_host: a box array's active region copied to the host
  active_region.py    active_region: the occupied bounds the box kernels run over
  box_owner.py        box_owner (Warp func): a box's cell, -1 empty or outside
  direction.py        direction (Warp func): the six face steps
  seam_box.py         seam_box (Warp func): a joint's box on a seam (another joint among its 26 neighbours)
  seam_flags.py       seam_flags: the seam boxes marked once per step for the substeps
  border_openness.py  border_openness (Warp func): 1 inside a cell, through a joint's seam down to 0 at the seal
  cell_sums.py        cell_sums: energy, size, place, heat, potential, want, compression, clock per cell
  uneven_field.py     uneven_field: seeded ripples at every size (FFT, numpy), the start's unevenness
  multigrid.py        Multigrid: the GPU Poisson solver, open-space boundary
  pull.py             pull: the A-energy's potential on a cube round the active region (CUDA graph)
  time_levels.py      time_levels: each cell's own substep (local clocks), boxes ordered by level
  on_site.py          on_site: each box's phase turned by the pull and the fluid's push (a joint's stiffer)
  hop.py              hop: exact two-box rotations across faces; friction at A-B faces, heat booked, spread planned
  fluid_energy.py     fluid_energy: the fluid equation's own energy (flow, pull, push), plain or clocked
  spread_heat.py      spread_heat: the energy friction took spreads into the boxes round the seam, exactly
  energy_flow.py      energy_flow: one step of the fluid on local clocks, Strang-split exact rotations
  local_time.py       local_time: each cell's clock reading advances by the proper time it lived
  joint_swing.py      joint_swing: each joint a spring with inertia, its aim for the moves
  box_moves.py        box_moves: 64 phases of plate moves (size, pull, push; wrapping, connected)
  size_correction.py  size_correction: the start's one correction, sizes follow energy
  joint_collapse.py   joint_collapse: a joint past its limit vanishes, its energy volumes fall together
  volume_split.py     volume_split: a volume past the split size tears, the tear a new joint
  step.py             step: sums, pull, energy flow, swing, plate moves, collapses, splits
  stats.py            stats: energy and drift, heat, cells, joints, collapses, splits, growth, lumps, seals, wrapping
  view_source.py      view_source: per occupied box its place, energy, kind, border flag (GPU)
"""
from physics.potts.settings import Settings
from physics.potts.universe import ENERGY_VOLUME, JOINT, PAD, Universe
from physics.potts.uneven_field import uneven_field
from physics.potts.multigrid import Multigrid
from physics.potts.new_universe import new_universe
from physics.potts.contact_pairs import contact_pairs
from physics.potts.touch_pairs import touch_pairs
from physics.potts.region_host import region_host
from physics.potts.active_region import active_region
from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.seam_box import seam_box
from physics.potts.seam_flags import seam_flags
from physics.potts.border_openness import border_openness
from physics.potts.cell_sums import cell_sums
from physics.potts.pull import pull
from physics.potts.time_levels import time_levels
from physics.potts.on_site import on_site
from physics.potts.hop import hop
from physics.potts.fluid_energy import fluid_energy
from physics.potts.spread_heat import spread_heat
from physics.potts.energy_flow import energy_flow
from physics.potts.local_time import local_time
from physics.potts.joint_swing import joint_swing
from physics.potts.box_moves import box_moves
from physics.potts.size_correction import size_correction
from physics.potts.joint_collapse import joint_collapse
from physics.potts.volume_split import volume_split
from physics.potts.step import step
from physics.potts.stats import stats
from physics.potts.view_source import view_source

__all__ = ["Settings", "ENERGY_VOLUME", "JOINT", "PAD", "Universe", "uneven_field", "Multigrid", "new_universe",
           "contact_pairs", "touch_pairs", "region_host", "active_region", "box_owner", "direction", "seam_box",
           "seam_flags", "border_openness", "cell_sums", "pull", "time_levels", "on_site", "hop", "fluid_energy",
           "spread_heat", "energy_flow", "local_time", "joint_swing", "box_moves", "size_correction", "joint_collapse", "volume_split", "step",
           "stats", "view_source"]
