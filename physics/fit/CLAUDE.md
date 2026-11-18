# physics/fit/

The fit search (bible `physics/quantum-relativity`, "How results are trusted"): settings
fitted on the expansion shape (slowing, then speeding up) and structure only, on the
engine rebuilt on the seven axioms. One directory per purpose:

- `measures/`: what the fit reads off a universe (size, observer clock, lumps, the gap between them).
- `score/`: the score of a run: hold x (expansion shape + structure) / 2.
- `search/`: one universe run and scored, the search's phases, the holders, the summary.
- `fitted/`: what the fit chose, apart from the defaults.

Each holds `definitions/` (what things are), `logic/` (what happens), `tests/` and its
own `CLAUDE.md`. `__init__.py` here lists the public names; `__main__.py` is the command.

## Must not get wrong

- **Nothing loose.** A directory holds subdirectories plus only `__init__.py`,
  `CLAUDE.md` (and `__main__.py` here). Definitions never import logic. `measures` and
  `score` never import `search` or `fitted`.
- **Fit set only.** Never put a test scene (orbit, bound lump, black hole, Hubble gap,
  timescape, two lumps) into the score.
- **Never change a default.** What the fit chose lives in `fitted/`; the owner decides
  about defaults.
- **Nothing is fitted, and nothing has been searched** on the rebuilt engine
  (2026-11-18). The 2026-11-17 search ran on the old engine: its numbers say nothing
  here and were deleted.
- **Marks and ranges are written down before a search.** Change one only before a
  search, and write down what the old one let through.
- **A short score can be a runaway caught midway.** Only the hold phase (four times
  longer) says a point stands. Never report a short score alone.
- **Local clocks, no dark energy.** No dark-energy, base-push, seal or drift setting is
  ever searched; the engine has none.
- **Ended and unfinished runs are counted**, never dropped.
- **Results are data**: OUT_DIR goes to tmp or object storage, never into the repo.
- **One search at a time**; the GPU is shared.

## Open, for the owner

- **The joints are most of the size.** The start lays every joint in balance; at the
  defaults a joint is 2.2 times a volume's length and the joints hold 76% of the energy.
- **Phase 1 (gravity) comes first** (bible, "The fit goes in phases"): its score is
  `scenes/gravity/definitions/pass_mark.py`, its ranges `search/definitions/gravity_ranges.py`.
  The expansion and structure search below waits for it.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.fit /tmp/fit-search      # whole search, hours
PYTHONPATH=. .venv/bin/pytest -q physics/fit                      # GPU tests skip without CUDA
```
