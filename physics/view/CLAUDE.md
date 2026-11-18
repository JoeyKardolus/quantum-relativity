# physics/view/

The only viewer: the grid of cells of `physics.engine`, every cell at its own size, in
a window or as saved pictures. Rebuilt 2026-11-18 on the rebuilt engine's public names.

One directory per purpose. Each holds `definitions/` (what things are), `logic/` (what
happens), `tests/` and its own `CLAUDE.md`:

- `layout/`: every cell in its place of the pattern; the cells' sizes warp the grid.
- `colours/`: red energy volumes, blue joints, opacity following packing.
- `drawing/`: the whole grid in three dimensions (the default) and the slice (key C).
- `source/`: the engine read as something to draw. The only place that imports it.
- `window/`: the pyglet window, its keys, the stepping thread.
- `pictures/`: PNG files saved while a source steps.

`__init__.py` holds the public names, `__main__.py` the command line, `tests/` the
whole viewer on a real engine run plus the helpers every test directory shares.

## Must not get wrong

- **Nothing loose.** A directory here holds subdirectories plus only `__init__.py`,
  `__main__.py` (here only) and `CLAUDE.md`.
- **Definitions never import logic.** One public function or type per file.
- **The cells are the grid; Planck boxes are never drawn or read.**
- **A cell never changes place**, so nothing flickers (`layout/`).
- **One fixed colour scale**, never stretched to a frame (`colours/`).
- **Headless by default.** `__init__.py` loads no OpenGL and no engine. Import
  `CellMesh`, `CellWindow`, `WholePicture` and `CellsSource` by module path.
- **Never tune a drawing number to make a run look good**, and never draw a cell the
  engine does not have.

## What the pictures show now

The engine has one size law for volumes and joints (more energy, smaller), and the start
lays every joint in balance: at the defaults a joint is about 2.2 times as long as an
energy volume (22 against 10). The grid is drawn that way. That is the engine's state,
not a drawing choice. `--scene two_lumps | coast | orbit | lone` opens a gravity scene
(`physics.scenes.gravity`) with the fitted phase 1 settings; a run steps the layers
below axiom 7 unless `--all-rules` is given.

## Not done

- After mass collapses cells fold and one layout takes seconds (`layout/CLAUDE.md`).
- No frame rates were measured on the rebuilt engine.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--scene NAME] [--all-rules] [--cells N] [--seed S] [--set name=value ...]
PYTHONPATH=. .venv/bin/python -m physics.view --pictures DIR --steps 40 --every 4 [--slice]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```

Tests that need CUDA or a display skip cleanly without them. Keys: `window/CLAUDE.md`.
