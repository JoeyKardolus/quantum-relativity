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
- **Nothing is fitted yet.** The file holds no values: no search has run on the cell-grid
  engine, and the earlier candidate named settings that no longer exist. `load_fitted()`
  then gives the default universe.
- **Space is measured inside** (owner, 2026-11-17): the universe wraps. `room` is every
  joint's length plus the lengths of the least packed volumes holding half the plus
  quanta: a sum of cell lengths, never a box count.
- **Nothing happening scores worst.** `fit_score` is 0 .. 1, every part a gain since the
  start: room's growth times its dip (slow, then fast); region contrast, lump share,
  sealed share, lump spread. Scales fixed, not fitted; they were set for the earlier
  engine and kept.
- **Local clocks, no dark energy.** Thirds are cut on the observer's time (clock rate
  weighted by quanta); the counter's score sits beside it (`counter`). No dark-energy or
  base-push setting is ever searched.
- **Searched** (`SEARCHED`, nine): pull_strength, light_speed, push_strength,
  squeeze_rate, crossing_rate (log), seal_thin, collapse_thin, tear_length (linear), the
  start's unevenness (0 .. 0.9). Ranges, score, size and seeds are fixed before a search.
- **One search at a time**: `workers` universes, a process each. Never start a second
  GPU job beside it.
- **Unfinished is not measured.** A run past `seconds`, or one the engine cannot step,
  scores 0 with its error; count them in every write-up: eventful universes step slowly
  (collapses and tears run on the host).
- **Known weakness**: a lump is a volume with twice the start's mean quanta, one cell; read
  Sr (region contrast) and `largest_lump` beside the score.
- **Results are data**: OUT_DIR goes to tmp or object storage, never into the repo.
- **Seeds**: judge a setting by its mean over seeds.
- **Gone**: the gradient fit through a smooth stand-in. It modelled the earlier engine's
  rules (boxes as space); a stand-in of the cell grid is not built.

## Measured (2026-11-17)

Only the two-point test search has run on the cell grid (8 places, 8 steps). The default
universe would score 0 on expansion: its size shrinks (`scenes/CLAUDE.md`).

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-search      # whole search, hours
PYTHONPATH=. .venv/bin/pytest -q physics/fit
```
