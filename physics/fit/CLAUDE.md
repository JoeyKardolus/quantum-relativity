# physics/fit/

The fit search (bible `physics/quantum-relativity`, "How results are trusted"): settings are
fitted on the fit set only, the expansion shape (slowing, then speeding up) and structure
(lumps, contrast, a spectrum that grows more at large scales). Module map in `__init__.py`.

## Must not get wrong

- **Fit set only.** Never put a test scene (orbit, bound lump, black hole, Hubble gap,
  timescape) into the score. A winner is a candidate until it passes those scenes, run unfitted from
  `physics.scenes` with criteria written before the run, on two runs.
- **Never change a default from here.** A setting that wins goes to the owner as a
  scorecard on the issue; the owner decides.
- **Local clocks, no dark energy.** The expansion dip is timed by the mean local clock of the
  most packed energy volumes (as an astronomer would), not the step counter; searches before
  2026-11-16's change used the counter. No dark-energy or base-push setting is ever searched.
- **Fixed before a run**: ranges, score, run size and seeds. Change them only before a new
  search, and say so in its write-up. The start's seed is not fitted.
- **One GPU run at a time**: the search runs every universe in one process, one after another.
  Never start a second GPU job beside it.
- **Results are data**: OUT_DIR goes to tmp or object storage, never into the repo.
- **Seeds**: the whole-quanta engine is bit-repeatable per seed; compare settings by their
  mean over seeds.

## Gradient fit (`stand_in_score`, `gradient_fit`, `compare_fits`)

- **Gradients go through a stand-in**, never the engine: whole quanta, claims and collapse
  have no gradient. The stand-in is the engine's fluid in float psi, every box keeping its
  kind, scored like `fit_run` without lumps and with the counter's time. A stand-in
  winner is checked on the real `fit_run` score.
- **Its gradient is a direction, not a derivative.** The full tape gradient blew up to 1e26
  in 10 steps (a box's own quanta turning its phase through the steep joint push), so
  potentials, openness and clocks read a density the tape does not see. Adam steps by a
  fixed share of the unit cube.
- Stand-in universes run side by side, each with its own settings; the engine takes one
  setting per batch.

## First comparison (2026-11-16, 7^3 cells of 10, seed 1 start, one run)

Stand-in 24 steps: random, 64 points side by side, best 0.42 in 2.8 s; gradient, 4 starts x
15 steps, best 0.16 (peak 0.28) in 14.6 s. Real score, seeds 1-3: defaults 0.00; random's
two best -0.10, -0.11; gradient's best 0.12 (pull 1e-6 and steepness 2.0, both range
edges), its second -0.13. The stand-in ranks the real score badly; one comparison only.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-search      # whole search, 140 runs of about 15 s
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-compare --compare   # gradient against random, about 5 min
PYTHONPATH=. .venv/bin/pytest -q physics/fit                     # tiny universes: search, gradient
```
