"""Scenes: the fit's scorecard (bible `physics/quantum-relativity`, "How results are trusted").

Each scene builds a start, runs the fluid engine (physics.fluid: new_grid, step, stats,
Settings; Grid.from_host for placed blobs) for a given number of steps and returns
measured numbers as a dict. No scene passes or fails, and none tunes a setting.

Module map:
  expansion.py       expansion: occupied and energy radius over time, growth rate, its trend
  hubble_gap.py      hubble_gap: early against late growth rate of the expansion scene
  structure.py       structure: the density contrast's clustering spectrum over time
  bound_lump.py      bound_lump: the biggest lump's size against its surroundings' size
  orbit.py           orbit: a small blob round a big one, radius, period, period^2 / radius^3
  black_hole.py      black_hole: sealed region radius against collapsed energy, per ball
  sampled_run.py     sampled_run: step a grid, measure it every few steps
  blob_start.py      blob_start: even moving balls of energy in empty space
  box_energies.py    box_energies: A- and B-energy per box on the host
  energy_centre.py   energy_centre: energy-weighted centre of masked boxes
  rms_radius.py      rms_radius: energy-weighted RMS distance from that centre
  growth_rates.py    growth_rates: d ln(size) / dt between samples
  power_spectrum.py  power_spectrum: shell power of the density contrast
  sealed_region.py   sealed_region: the sealed boxes round the most squeezed one
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

__all__ = ["expansion", "hubble_gap", "structure", "bound_lump", "orbit", "black_hole", "sampled_run",
           "blob_start", "box_energies", "energy_centre", "rms_radius", "growth_rates", "power_spectrum",
           "sealed_region"]
