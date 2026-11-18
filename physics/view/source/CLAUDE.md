# physics/view/source/

The engine read as something to draw. The only directory in the viewer that imports
`physics.engine`, and only its public names.

- `definitions/`: the source interface and the names of the counts the title shows.
- `logic/`: the engine's grid as pieces, its counts, and `CellsSource`.
- `tests/`: real engine runs with collapses and with tears; every living cell is drawn
  once and never changes slot.

## Must not get wrong

- **Read, never write.** Nothing here changes the engine's state except `step()` and
  `reset()`, which call the engine's own `step` and `new_universe`.
- **No box is read.** Six per-cell arrays are copied to the host per frame: kind,
  length, packing, torn_from, made_kind, swirl_axis.
- **Settings come from the engine.** `--set name=value` becomes `Settings(**overrides)`;
  the start's `unevenness` and `slope` are settings too.
- Tests here need CUDA and skip cleanly without it.

## Assumed

- `Grid.host(name)` is the way to read a per-cell array. `read_cells` does not hand
  out `packing`, `made_kind` or `swirl_axis`, which the drawing needs.

## Not done

- The engine has no lump count any more, so the title shows `warp` (from
  `measure_grid`) instead of `lumps`.
