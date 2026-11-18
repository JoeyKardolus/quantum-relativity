# physics/view/

The only viewer: the grid of cells of `physics.engine` as a perfect A B A B grid that
only lumps warp, in a window or as saved pictures.

One directory per purpose. Each holds `definitions/` (what things are), `logic/` (what
happens), `tests/` and its own `CLAUDE.md`:

- `layout/`: every cell in its place of the pattern; smaller cells warp the grid
  round them, nothing else moves.
- `colours/`: red energy volumes, blue joints; fills clear at the background's packing
  and solid in a lump; outlines always.
- `drawing/`: the slice (the default) and the whole grid in three dimensions (key C).
- `source/`: the engine read as something to draw. The only place that imports it.
- `window/`: the pyglet window, its keys, the stepping thread.
- `pictures/`: PNG files saved while a source steps.

`__init__.py` holds the public names, `__main__.py` the command line, `tests/` the
whole viewer on a real run plus the shared test helpers.

## Must not get wrong

- **Nothing loose.** A directory here holds subdirectories plus only `__init__.py`,
  `__main__.py` (here only) and `CLAUDE.md`.
- **Definitions never import logic.** One public function or type per file.
- **The cells are the grid; Planck boxes are never drawn or read.**
- **A cell never changes place**, and a corner out of reach of every change never
  moves: no fit, no step that looks at the whole frame (`layout/`).
- **The pattern always shows**: every drawn cell has its outline (`drawing/`).
- **Light at any size.** One frame lays out and draws one layer, plus the packed cells
  for the whole grid.
- **One fixed colour scale**, never stretched to a frame (`colours/`).
- **Headless by default.** `__init__.py` loads no OpenGL and no engine. Import
  `CellMesh`, `CellWindow`, `WholePicture` and `CellsSource` by module path.
- **Never tune a drawing number to make a run look good**, and never draw a cell the
  engine does not have.

## What the pictures show now

At the defaults a joint is about 2.2 times as long as an energy volume (22 against 10).
The perfect grid is drawn from those two typical lengths (the medians), and the picture
follows its frame, so the grid sits still on screen while the universe's size changes;
the title carries the size. `--scene two_lumps | coast | orbit | lone` opens a gravity scene
(`physics.scenes.gravity`) with the fitted phase 1 settings; a run steps the layers
below axiom 7 unless `--all-rules` is given.

## Not done

- Sizes are shown against the neighbourhood and cut off at 4 times typical
  (`layout/CLAUDE.md`); the packing, in the colour, is exact.
- Frame rates: synthetic cells only (`window/CLAUDE.md`).

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--scene NAME] [--all-rules] [--cells N] [--seed S] [--set name=value ...]
PYTHONPATH=. .venv/bin/python -m physics.view --pictures DIR --steps 40 --every 4 [--whole]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```

Tests that need CUDA or a display skip cleanly without them. Keys: `window/CLAUDE.md`.
