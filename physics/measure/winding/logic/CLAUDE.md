# Measure: winding, logic

What happens when the winding is read. Each file's docstring is its contract.

- `phase_winding.py`: the whole turns round every unit square of boxes in an array, for
  each of the three normals.
- `cell_turns.py`: one cell's block in, one whole number per layer out: that layer's
  squares added up, which is the turns round the layer's rim when no box is empty.

## Must not get wrong

- **`cell_turns` takes the engine's own arrays**: whole signed quanta and a phase in
  radians. Minus quanta (a joint) read the same as plus.
- **Adding the squares of a layer gives the rim** because every inner edge is walked once
  each way and cancels. That stops being true where a box is empty: the squares touching
  it are dropped.
- **Ask about the cell's own axis.** Across the other two axes a whole swirl reads 0.
- **`tests/wound_block.py` is a helper, not a test**: a block wound a known number of turns.
- Logic imports definitions and nothing from the engine.
