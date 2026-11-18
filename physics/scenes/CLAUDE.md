# physics/scenes/

The scorecard (bible `physics/quantum-relativity`): each scene builds a start, runs the
engine and returns numbers. Map in `__init__.py`. Rebuilt 2026-11-18 on the rebuilt
engine's public names, in the engine's layout.

## Layout

One directory per scene (12) and per shared piece (`run`, `placed_start`, `volumes`,
`along_grid`). Each holds `CLAUDE.md`, `__init__.py`, `tests/`, `definitions/` (what
things are: constants, pass marks, formulas) and `logic/` (what happens: build, step,
read), each of those two with its own `CLAUDE.md` and `tests/`. Nothing else is loose;
`tests/test_the_package_keeps_its_layout.py` checks it.

## Must not get wrong

- **Numbers, no verdict.** A scene returns what it measured beside the known value. Its
  pass mark is `definitions/pass_mark.py`, written before any run and applied from
  outside. Never change a default, seed, start or mark to make a scene look better; a
  mark changed with the engine carries its date and reason.
- **Definitions never import logic**, and never touch a universe.
- **Cells only.** A scene reads quanta, lengths, clocks and who has a way to whom,
  through the engine's public names. Only `spin` reads a block: a swirl lives in a cell.
- **Size is the sum of cell lengths**; the universe wraps. Distance is summed lengths.
- **All rules on, always.** The rebuilt engine has no off switch, so no scene has a
  side-by-side run with a rule off. A twin run differs in its start only.
- **Placed starts** (orbit, black hole, two lumps, isotropy) break "nothing placed by
  hand" on purpose, through `placed_start`. Every scene takes `start`, a candidate's
  start amounts (settings names).
- **A pull is read beside its law in the same wrapping box**, never bare, and the twin
  run without the lump is subtracted: an even grid moves energy and tilts the pull by itself.
- **A slow pull has not arrived**, and a packed core's clock is slow.

## Deleted with the old engine

`sealed_region` (no seal in the engine); the staged quantum doors; hand-wound whirls and
the double whirl; every crossing-off run; the scenes' own `unevenness` argument (it is
a setting); `stats` readings (now `measure_grid`, `read_books`, `count_lumps`).

## Not done

- **No scene has been run at its scorecard size on the rebuilt engine.** The numbers of
  2026-11-17 (2 of 12) were the old engine's and are gone from this file.
- Tests run on the engine's `held` settings (`tests/_gpu.py`): on the defaults a
  universe of 8 a side falls into one volume within 30 steps.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/scenes     # needs CUDA; under a minute on a free card
```
