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
- **Not bit-repeatable**: the same seed run twice moves the score by up to 0.04; compare
  settings by their mean over seeds against that noise.

## First search (2026-11-16, 8^3 cells of 6^3, 120 steps, 140 runs, 0 errors)

Defaults -0.41 +- 0.03 (5 runs); best (attraction 0.047, push 0.45, push_power 1, collapse
8.9, seal 2.9, friction 0.33) 0.17 +- 0.06 over seeds 1-3, all on structure; attraction
drives it (rank correlation 0.82). No run slows then speeds up (dip -0.27 to -0.01). The top
three and defaults, tested unfitted: no setting passes any scene on both runs (orbits fall
in, nothing seals, Hubble late/early 0.05 to 0.2 against 1.09). Scorecard on issue #1.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-search      # whole search, about 45 min on the 5080
PYTHONPATH=. .venv/bin/pytest -q physics/fit                     # two points on a tiny universe
```
