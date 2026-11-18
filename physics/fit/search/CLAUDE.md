# The search

Many universes over the searched settings, one JSON line per run.

- `definitions/`: the ranges, the points, the plan, when a universe is over, when a point holds.
- `logic/`: one universe run and scored, runs side by side, the phases, the holders, the summary.
- `tests/`: the whole search on tiny universes (needs the GPU).

Phases: defaults, random (Latin hypercube), refine rounds, hold (the best calm points
run four times longer), sweep (one setting moved at a time).

## Must not get wrong

- **Twelve settings are searched** (`SEARCHED`): every rule's strength and the start's
  energies. Ranges were fixed on 2026-11-18, before any run on the rebuilt engine.
- **A point is fitted only if it holds**: half its short score kept in the long run,
  hold factor above 0.9.
- **A process per universe.** A run that raises, hangs or dies is recorded with score 0
  and its error; it takes nothing with it and is never dropped.
- **Resumable.** A run whose id is in the results file is never run again; `minutes`
  stops starting new runs.
- **Results go outside the repo.**
- **The GPU is shared**: `workers` universes fit side by side only while the card has
  room; one search at a time.

## Not done

- No search has run on the rebuilt engine, so the speed (universes a second) and the
  wall-time cap of 240 s a short run are the old engine's and unmeasured here.
- The push range's top (2e3) was set wide because the engine's own test runs need a
  push of 40 for the start's joints to hold; the default (2e-6) collapses at once.
