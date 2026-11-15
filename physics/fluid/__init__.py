"""The fluid engine: the foam's energy as a fluid on a fine fixed grid of boxes on the GPU (Warp).

Bible `physics/quantum-relativity`, "How the engine holds the foam": every box holds
pulling (A) and pushing (B) energy and how each flows; energy volumes and joints are
regions of the fluid, the ABAB pattern lives below the box size. Nothing is a ball or a link.

Module map:
  settings.py        Settings: every rule's strength, all_on() defaults
  grid.py            Grid: energies (whole quanta, exact), momenta, potential, clock, heat, start references
  local_length.py    local_length: a box's measured length, start mean energy / its energy (rule 4)
  uneven_field.py    uneven_field: a Gaussian random field with power spectrum k**slope (FFT)
  new_grid.py        new_grid: space filled with uneven energy at rest, in an empty margin
  b_pressure.py      b_pressure (Warp func): steep B push that stops rising past the collapse limit
  b_wave_speed.py    b_wave_speed (Warp func): sqrt(dP/db), 0 past the collapse limit
  flow_velocity.py   flow_velocity (Warp func): momentum / energy, 0 in an empty box
  multigrid.py       Multigrid: GPU Poisson V-cycles with an open-space monopole boundary
  pull_potential.py  pull_potential: the A-energy's gravity-like potential
  sending_speed.py   sending_speed (Warp func): |u| along an axis, 0 into a closed wall
  clock_rate.py      clock_rate: each box's own time per step from its concentration, joints included
  joint_openness.py  joint_openness (Warp func): 1 relaxed, straight down to 0 at the seal point
  a_passage.py       a_passage (Warp func): A flows only where its box holds its joint share of B
  fastest_wave.py    fastest_wave: the CFL wave-speed sum
  pull_kick.py       pull_kick: the pull accelerates A's flow
  push_kick.py       push_kick: B's pressure accelerates B's flow
  transport.py       transport: donor-cell fluxes of energy, momentum and heat, in each box's time
  rub.py             rub: A-through-B friction by B compression, the seal lock; the loss is heat
  leak.py            leak: A through open joints to the neighbours by their pull (drift)
  step.py            step: one step of CFL substeps, in place, every box in its own time
  sealed_radius.py   sealed_radius: the largest sealed region's measured radius (black hole)
  proper_distance.py proper_distance: path lengths from one box along the structure (host)
  dimension.py       dimension: how measured volume grows with proper radius (host)
  stats.py           stats: energy, drift, sizes, lumps, seals, cores, clocks, heat, motion, pull
The viewer (physics/view) reads grid.a_energy and grid.b_energy (int64 quanta; the
viewer's opacity is relative to the start, so the unit cancels) and gathers on the GPU.
"""
from physics.fluid.settings import Settings
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.uneven_field import uneven_field
from physics.fluid.new_grid import new_grid
from physics.fluid.multigrid import Multigrid
from physics.fluid.pull_potential import pull_potential
from physics.fluid.fastest_wave import fastest_wave
from physics.fluid.pull_kick import pull_kick
from physics.fluid.push_kick import push_kick
from physics.fluid.transport import transport
from physics.fluid.rub import rub
from physics.fluid.clock_rate import clock_rate
from physics.fluid.leak import leak
from physics.fluid.local_length import local_length
from physics.fluid.sealed_radius import sealed_radius
from physics.fluid.proper_distance import proper_distance
from physics.fluid.dimension import dimension
from physics.fluid.step import step
from physics.fluid.stats import stats

__all__ = ["Settings", "QUANTUM", "Grid", "uneven_field", "new_grid", "Multigrid", "pull_potential", "fastest_wave",
           "pull_kick", "push_kick", "transport", "rub", "clock_rate", "leak", "local_length", "sealed_radius",
           "proper_distance", "dimension", "step", "stats"]
