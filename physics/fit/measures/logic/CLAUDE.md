# Measures, logic

What happens when the fit measures. Each file's docstring is its contract.

- `read_universe.py`: copies the cells, the ways and the counts of one moment to the host.
- `measure_cells.py`: size, observer rate, alive, spent, lumps, region contrast.
- `take_snapshot.py`: the cells of one moment, kept to follow lumps back.
- `gap_between_lumps.py`: the distance between the end's lumps at every snapshot.
- `held_share.py`: the energy in lumps two thirds through that is still in lumps at the end.

## Must not get wrong

- **Nothing here writes to the universe.**
- **A cell id is the same cell for good**, so a row means the same cell in every read.
  The gap and the held share rest on that.
- **Ways are read again only after a collapse or a tear** (`version` changed); reading
  them is the slow part.
- **`measure_cells` raises when the volumes hold nothing.** Do not turn that into a 0.
- **Tests use hand-made grids** (`tests/hand_cells.py`) whose every number is known;
  only `read_universe` needs the GPU.
