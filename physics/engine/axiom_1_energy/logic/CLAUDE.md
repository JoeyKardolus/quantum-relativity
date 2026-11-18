# Axiom 1: energy, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `uneven_field.py`, `start_energy.py`: the start says energies and nothing else. The field has mean 0 and the asked spread at every size.
- `clear_ledgers.py`: a step starts with empty per-step counts.
- `flow_inside.py`: the fluid moves between the boxes of one cell's block, six calls a step, in whole quanta, on the cell's local time.
- `count_cells.py`: the only writer of `cell_quanta`.
- `read_books.py`: every box summed in whole numbers beside the counts that changed the total.

Energy moving from cell to cell is `cross` (axiom 4): the same `pair_rotation`, taken by two boxes that face each other across a joint.

## Must not get wrong

- **No pair leaves its block** in `flow_inside`, and no thread reads a box of another cell.
- **Nothing here drives the fluid.** No pressure step, no turn of a phase: the fluid moves only because its phases differ from box to box. What makes them differ is the swirl each cell starts with (axiom 2), what arrives (axiom 4) and the pull (axiom 5).
- **A fluid of one phase stands still**, at the block's sides too. A test pins it: 40 steps, not one quantum moves. Whole quanta are rounded by a draw, so once anything moves there is noise: one moved quantum becomes about 2,000 quanta changing box a step in each cell and stays there.
- **Phases are float64.** In float32 a slow cell's small turns were thrown away and its fluid froze.
- **An empty box and a dead id do nothing**; an empty box can be filled.
- **Books.** Every change of a total is one whole number written where the change is made. `read_books` must give `book_law` all zeros after every step.

## Tests

`tests/conftest.py`: `small_grid` (a 4-a-side start lattice on the CPU), `inner_step` (the six flow calls on closed cells).
