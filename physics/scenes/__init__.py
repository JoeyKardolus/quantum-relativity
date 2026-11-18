"""Scenes: the scorecard (bible `physics/quantum-relativity`, "How results are trusted").

Each scene builds a start, runs the engine (physics.engine) with every rule on and
returns measured numbers as a dict, beside the known value. No scene passes or fails
itself and none tunes a setting: its pass mark, written before any run, lives in its
own `definitions/pass_mark.py` and is applied from outside. Everything is read from the
cells: their quanta, their lengths, their clocks and who has a way to whom. Planck
boxes are not space; only `spin` reads a cell's own block.

One directory per scene of the scorecard:
  orbit/          does extra energy beside a fuller volume go round it (Kepler)
  bound_lump/     does the fullest volume keep its size while the rest of space changes
  black_hole/     how far round a packed core nothing gets away: the affected region, not the core
  hubble_gap/     early against late growth rate of the universe
  timescape/      growth timed by the counter and by the fullest and emptiest volumes' clocks
  expansion/      the universe's size over time, its growth rate and its trend
  structure/      the clustering spectrum of the volumes' energy, and its lumps
  dimension/      ball and spectral dimension of the volume graph, beside a plain grid's
  isotropy/       pull and arrived energy round a lump, on an axis against off it
  spin/           does a cell stay the one whole swirl it was born as
  quantum_split/  do splits settle whole, in each cell as often as its share
  two_lumps/      the pull between two lumps against distance, beside one over distance squared
  gravity/        phase 1 of the fit: stand, fall, coast, fade with distance, go round
and the pieces they share:
  run/            a candidate's start in the settings; step and measure every few steps
  placed_start/   an even grid with chosen volumes packed
  volumes/        the volumes as a field, as lumps and as observers; the power spectrum
  along_grid/     distances as summed lengths, the pull on the volumes, the law beside it
Each holds `definitions/` (what things are: records, constants, pass marks, formulas),
`logic/` (what happens), `tests/` and its own `CLAUDE.md`. This file holds only the
public names.
"""
from physics.scenes.along_grid import box_depth, box_field, log_slope, middle_distances, row_cells, volume_pull
from physics.scenes.black_hole import black_hole
from physics.scenes.bound_lump import bound_lump
from physics.scenes.dimension import space_dimension
from physics.scenes.expansion import expansion, growth_rates
from physics.scenes.gravity import gravity
from physics.scenes.hubble_gap import hubble_gap
from physics.scenes.isotropy import isotropy
from physics.scenes.orbit import orbit
from physics.scenes.placed_start import packed_start
from physics.scenes.quantum_split import quantum_split
from physics.scenes.run import even_volume_quanta, sampled_run, start_settings
from physics.scenes.spin import spin
from physics.scenes.structure import structure
from physics.scenes.timescape import deceleration, timescape
from physics.scenes.two_lumps import two_lumps
from physics.scenes.volumes import LUMP_FACTOR, count_lumps, observer_cells, power_spectrum, volume_field

SCENES = {"orbit": orbit, "bound_lump": bound_lump, "black_hole": black_hole, "hubble_gap": hubble_gap,
          "timescape": timescape, "expansion": expansion, "structure": structure, "dimension": space_dimension,
          "isotropy": isotropy, "spin": spin, "quantum_split": quantum_split, "two_lumps": two_lumps,
          "gravity": gravity}

__all__ = ["LUMP_FACTOR", "SCENES", "black_hole", "bound_lump", "box_depth", "box_field", "count_lumps", "deceleration",
           "even_volume_quanta", "expansion", "growth_rates", "hubble_gap", "isotropy", "log_slope", "middle_distances",
           "observer_cells", "orbit", "packed_start", "power_spectrum", "quantum_split", "row_cells", "sampled_run",
           "space_dimension", "spin", "start_settings", "structure", "timescape", "two_lumps", "volume_field",
           "volume_pull"]
