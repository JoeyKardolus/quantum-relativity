# physics/view/source/logic/

What happens when the engine is read. Everything here imports the engine (Warp).

- `grid_pieces.py`: the engine's `Grid` to `Pieces`: host copies of six per-cell
  arrays, then `torn_lines`, `cell_pieces` for one layer or for the `packed_cells`,
  and `relative_packing` of the pieces only.
- `cell_stats.py`: counts for the title from the engine's `measure_grid`, `read_books`
  and its `collapses` and `tears`.
- `cells_source.py`: `CellsSource`, the engine behind the `Source` interface.

## Must not get wrong

- `reset()` takes the start's packing marks and size. Every later frame is read
  against them.
- An unknown setting raises `TypeError`, a refused value `ValueError`, both when the
  source is made. The command line turns them into a message.
- `cell_stats` runs `read_books`, which counts every box on the GPU. The stepper
  calls it at most twice a second.
- `tests/` runs the engine at 8 cells a side, seed 1, and checks the viewer's kind
  numbers against the engine's.
