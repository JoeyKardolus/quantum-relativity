"""Scenes: the fit's scorecard (bible `physics/quantum-relativity`, "How results are trusted").

Each scene builds a start, runs the cell engine (physics.potts: new_universe, step,
stats, Settings; Universe.from_host for placed blobs and placed phases) for a given number of steps and
returns measured numbers as a dict. No scene passes or fails, and none tunes a setting.

Module map:
  expansion.py       expansion: occupied (space) and energy radius over time, growth rate, its trend
  hubble_gap.py      hubble_gap: early against late growth rate of the expansion scene
  structure.py       structure: the density contrast's clustering spectrum over time
  bound_lump.py      bound_lump: the biggest lump's size against its surroundings' size
  orbit.py           orbit: a small blob round a big one, radius, period, period^2 / radius^3
  black_hole.py      black_hole: sealed region radius against collapsed energy, per ball
  sampled_run.py     sampled_run: step a universe, measure it every few steps
  blob_start.py      blob_start: moving balls of extra energy in an even checkerboard
  box_energies.py    box_energies: A- and B-energy per Planck box on the host, by owner kind
  energy_centre.py   energy_centre: energy-weighted centre of masked boxes
  rms_radius.py      rms_radius: energy-weighted RMS distance from that centre
  growth_rates.py    growth_rates: d ln(size) / dt between samples
  power_spectrum.py  power_spectrum: shell power of the density contrast
  sealed_region.py   sealed_region: the densest energy volume, its joints, and whether they seal it
  wave_field.py      wave_field: the fluid's complex number per box (psi) on the host
  space_dimension.py space_dimension: ball and spectral dimension of the cell-contact graph, beside a grid's
  isotropy.py        isotropy: pull and flow along a grid axis against the diagonals
  spin.py            spin: one whole-step whirl carried across a joint; winding, flow, strength, axis per step
  belt_trick.py      belt_trick: a whirl tied into the joint web turned once and twice; the web's twist per step
  timescape.py       timescape: growth of space timed by the step counter and by clocks in the densest lumps
  clocked_run.py     clocked_run: sampled_run that also keeps every cell's elapsed local time
  observer_cells.py  observer_cells: the most packed energy volumes that kept a clock all run
  deceleration.py    deceleration: q = -a'' a / a'^2 from a quadratic fit (q < 0 speeds up)
"""
from physics.scenes.expansion import expansion
from physics.scenes.hubble_gap import hubble_gap
from physics.scenes.structure import structure
from physics.scenes.bound_lump import bound_lump
from physics.scenes.orbit import orbit
from physics.scenes.black_hole import black_hole
from physics.scenes.sampled_run import sampled_run
from physics.scenes.blob_start import blob_start
from physics.scenes.box_energies import box_energies
from physics.scenes.energy_centre import energy_centre
from physics.scenes.rms_radius import rms_radius
from physics.scenes.growth_rates import growth_rates
from physics.scenes.power_spectrum import power_spectrum
from physics.scenes.sealed_region import sealed_region
from physics.scenes.wave_field import wave_field
from physics.scenes.space_dimension import space_dimension
from physics.scenes.isotropy import isotropy
from physics.scenes.spin import spin
from physics.scenes.belt_trick import belt_trick
from physics.scenes.clocked_run import clocked_run
from physics.scenes.observer_cells import observer_cells
from physics.scenes.deceleration import deceleration
from physics.scenes.timescape import timescape

__all__ = ["expansion", "hubble_gap", "structure", "bound_lump", "orbit", "black_hole", "sampled_run",
           "blob_start", "box_energies", "energy_centre", "rms_radius", "growth_rates", "power_spectrum",
           "sealed_region", "wave_field", "space_dimension", "isotropy", "spin", "belt_trick",
           "clocked_run", "observer_cells", "deceleration", "timescape"]
