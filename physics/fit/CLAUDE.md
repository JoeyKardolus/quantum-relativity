# physics/fit/

The fit search (bible `physics/quantum-relativity`, "How results are trusted"): settings
fitted on the expansion shape (slowing, then speeding up) and structure only. Module map
in `__init__.py`.

## Must not get wrong

- **Fit set only.** Never put a test scene (orbit, bound lump, black hole, Hubble gap,
  timescape) into the score. A winner is a candidate until it passes those, unfitted,
  with criteria written before the run.
- **Never change a default.** What the fit chose lives in `fitted/settings.json`
  (`load_fitted()`); the owner decides about defaults.
- **Space is measured inside, never at an edge** (owner, 2026-11-17). `fit_run` wraps;
  `room` is every joint's boxes plus those of the least packed volumes holding half the
  plus quanta. The box count is fixed, so room grows only as lumps pack; whole-space
  growth cannot show.
- **Nothing happening scores worst.** `fit_score` is 0 .. 1, every part a gain since
  the start: room's growth times its dip (slow, then fast); region contrast, lump share,
  sealed share, lump spread. Scales fixed, not fitted.
- **Local clocks, no dark energy.** Thirds are cut on the observer's time (clock rate
  weighted by quanta); the counter's score sits beside it (`counter`). No dark-energy or
  base-push setting is ever searched.
- **Searched** (`SEARCHED`, ten): pull_strength, light_speed, push, collapse_packing,
  crossing_rate, seal_squeeze (log), push_steepness, split_across (linear), the start's
  unevenness and joint_unevenness (0 .. 0.9). Ranges, score, size and seeds are fixed
  before a search.
- **One search at a time**: `workers` universes side by side, a process each. Never
  start a second GPU job beside it.
- **Unfinished is not measured.** A run past `seconds`, or one the engine cannot step,
  scores 0 with its error; count them in every write-up: eventful universes step slowly.
- **Known weakness**: uniform packing counts as lumps (the threshold is the start's
  packing) and swelling joints as room; read Sr (region contrast) beside the score.
- **Results are data**: OUT_DIR goes to tmp or object storage, never into the repo.
- **Seeds**: judge a setting by its mean over seeds; the best seed-1 point of
  2026-11-17 died on seeds 2 and 3.

## Gradient fit (`stand_in_score`, `gradient_fit`, `compare_fits`)

- **Gradients go through a stand-in**, never the engine (float psi, the six `NAMES`).
  It still scores the old way (occupied radius, by the counter): its winners say nothing
  until re-scored by `fit_run`. Its gradient is a direction, not a derivative.

## Measured (2026-11-17, RTX 5080)

24 places of 10 (1,728 volumes, 2.3M boxes, wrapping), 3,000 steps, 8 at once,
4.5 h: 105 runs, 51 measured, 37 died in `joint_swing`, 17 unfinished. **Void**: the
swing had three faults (fixed that day), and the best point (`fitted/`, 0.387
against the defaults' 0.253) won on them: its room was joints swelling. 22% of
its joints start past the collapse limit and collapse at once: it falls in
(24 places: 1,728 volumes to one in 4 steps). Search again (issue #1).

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-search      # whole search, hours
PYTHONPATH=. .venv/bin/pytest -q physics/fit
```
