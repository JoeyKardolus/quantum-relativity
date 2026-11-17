# physics/fit/

The fit search (bible `physics/quantum-relativity`, "How results are trusted"): settings
fitted on the expansion shape (slowing, then speeding up) and structure only. Module map
in `__init__.py`.

## Must not get wrong

- **Fit set only.** Never put a test scene (orbit, bound lump, black hole, Hubble gap,
  timescape, two lumps) into the score.
- **Never change a default.** What the fit chose lives in `fitted/settings.json`
  (`load_fitted()`); the owner decides about defaults.
- **Nothing is fitted** (2026-11-17): no searched point holds, so the file has no values
  and `load_fitted()` gives the default universe. Its `fitted_on` names the best
  candidate and what ends it.
- **The score is `fit_score`'s docstring**: hold x (expansion + structure) / 2. Nothing
  happening scores 0; fallen into one cell or torn without end, about 0; no cliff; tanh
  caps every part. Change it only before a search, and write down what the old one let
  through.
- **Space is measured inside.** Size is the sum of cell lengths; the gap between lumps
  sums the same cells in every sample, chosen by fewest cells, never by length.
- **A short score can be a runaway caught midway.** Only the hold phase (four times
  longer, `fit_holders`) says a point stands. Never report a 5,000-step score alone.
- **Local clocks, no dark energy.** Thirds are cut on the observer's time. No
  dark-energy or base-push setting is ever searched.
- **Searched** (`SEARCHED`, fifteen): every rule's strength and the start. Ranges are
  fixed before a search.
- **The GPU gives about 0.3 universes a second** (24 places, 5,000 steps) whatever the
  workers; `--minutes` runs a slice, the next call resumes. One search at a time.
- **Ended and unfinished runs are counted**, never dropped.
- **Results are data**: OUT_DIR goes to tmp or object storage, never into the repo.

## Measured (2026-11-17, 24 places, 2,332 universes)

Defaults score 0 on six seeds (size -0.6%). Of 1,500 random points 482 do nothing, 423
fall into under 5% of their volumes, 313 tear without end, 229 end spent, 21 stay calm
with structure. Three refine rounds took the best 5,000-step score from 0.215 to 0.615
(57 lumps holding 72% of the energy, size +28%, slowly, then fast). Hold phase, the 100
best calm points (0.25 to 0.62) for 20,000 steps: none holds; 68 tear without end, 30
end spent, half of them by step 7,968; the best tears 250 steps after its short run
stops. What stops it: the volumes round a lump empty without limit and an emptied
volume is enormous by the size law, or tears; no lump's border seals (0 of 108 joints,
27 of 48 inside it); heat only rises, so a weaker pull's lumps melt. The best points
sit at range edges (clock_packing, push_strength, squeeze_rate, the joints' unevenness
and length): the joints' rules as good as off. The sweep cannot decide collapse_thin
above 3.3 or tear_length above 1,000.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-search      # whole search, hours
PYTHONPATH=. .venv/bin/pytest -q physics/fit
```
