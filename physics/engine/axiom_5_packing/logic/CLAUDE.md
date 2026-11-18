# Axiom 5: packing, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `start_pull.py`: the pull starts at rest for the start's energy, found by sweeps; clocks at 0.
- `tick_clocks.py`: each cell's clock goes on by its local time.
- `spread_pull.py`: one wave step of every cell from its own neighbours.
- `pull_fluid.py`: the pull turns the fluid of every volume, box by box; the one place plus energy feels the pull.
- `pack_cells.py`: packing and clock rate from quanta and length.
- `press_joints.py`: each way's press; each joint's hardest way, press and squeeze.

## Must not get wrong

- **Who writes `pull`**: `start_pull`, `spread_pull`, and a tear (axiom 6), which gives its two new cells the torn volume's value.
- **`spread_pull` works nothing out over the whole grid.** It used to take the mean change off every cell each step; that reached every cell at once. Now the level all cells share sinks steadily and nobody reads it. Only `start_pull` levels the field, once, before step 1.
- **`pull_fluid` turns every box**, by the pull where the box stands, on the cell's local time. Toward a deeper pull the fluid falls behind and flows there. `pull_felt` is the largest turn: above pi it has wrapped and its direction is lost.
- **A joint's press is its hardest way**, never a sum. A press below 0 is kept; only the squeeze stops at 0.
- **The wave is taken from nearest neighbours**: a trace of a change runs one neighbour a step, ahead of light. See the axiom's overlay.

## Tests

`tests/` holds a wave test that needs CUDA (2,600 steps of two universes); its light-speed cases fail and are left failing.
