"""Scenes: the fit's scorecard (bible `physics/quantum-relativity`, "How results are trusted").

Each scene builds a start, runs the engine (physics.engine: new_universe, step, stats,
way_list, Settings, State) for a given number of steps and returns measured numbers as a
dict. No scene passes or fails, and none tunes a setting. Everything is read from the
cells: their quanta, their lengths, their clocks and who neighbours whom. Planck boxes
are not space and no scene reads one.

Module map:
  expansion.py       expansion: the universe's size (sum of cell lengths) over time, growth rate, its trend
  hubble_gap.py      hubble_gap: early against late growth rate of the expansion scene
  structure.py       structure: the clustering spectrum of the volumes' energy over time
  bound_lump.py      bound_lump: the most packed volume's length against the other volumes'
  orbit.py           orbit: extra energy beside a packed volume; how long it holds, where it goes
  black_hole.py      black_hole: how far round a packed core energy flows inward and the pull's depth reaches, against the core's energy
  two_lumps.py       two_lumps: the pull each of two placed lumps feels toward the other against distance, beside one over distance squared
  isotropy.py        isotropy: pull and arrived energy round one placed lump, on an axis against off it at the same distance
  spin.py            spin: a whirl placed inside one cell, its whole turns read at every step
  quantum_split.py   quantum_split: staged splits, the share of draws that settle on each side
  timescape.py       timescape: growth timed by the counter, by the densest and by the emptiest volumes' clocks
  space_dimension.py space_dimension: ball and spectral dimension of the volume-neighbour graph, beside a grid's
  sampled_run.py     sampled_run: step a universe, measure it every few steps
  packed_start.py    packed_start: an even grid with chosen volumes packed (placed starts)
  volume_field.py    volume_field: the volumes' quanta on their own lattice
  sealed_region.py   sealed_region: the region round the most packed volume that sealed joints close off
  observer_cells.py  observer_cells: the most packed energy volumes
  growth_rates.py    growth_rates: d ln(size) / dt between samples
  power_spectrum.py  power_spectrum: shell power of an energy contrast
  deceleration.py    deceleration: q = -a'' a / a'^2 from a quadratic fit (q < 0 speeds up)
"""
from physics.scenes.black_hole import black_hole
from physics.scenes.bound_lump import bound_lump
from physics.scenes.deceleration import deceleration
from physics.scenes.expansion import expansion
from physics.scenes.growth_rates import growth_rates
from physics.scenes.hubble_gap import hubble_gap
from physics.scenes.isotropy import isotropy
from physics.scenes.observer_cells import observer_cells
from physics.scenes.orbit import orbit
from physics.scenes.packed_start import packed_start
from physics.scenes.power_spectrum import power_spectrum
from physics.scenes.quantum_split import quantum_split
from physics.scenes.sampled_run import sampled_run
from physics.scenes.sealed_region import sealed_region
from physics.scenes.space_dimension import space_dimension
from physics.scenes.spin import spin
from physics.scenes.structure import structure
from physics.scenes.timescape import timescape
from physics.scenes.two_lumps import two_lumps
from physics.scenes.volume_field import volume_field

__all__ = ["black_hole", "bound_lump", "deceleration", "expansion", "growth_rates", "hubble_gap", "isotropy",
           "observer_cells", "orbit", "packed_start", "power_spectrum", "quantum_split", "sampled_run", "sealed_region",
           "space_dimension", "spin", "structure", "timescape", "two_lumps", "volume_field"]
