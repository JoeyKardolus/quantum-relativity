# Axiom 3: structure, definitions

What things are under this axiom. Each file's docstring is its contract.

- `kinds.py`: energy volume, joint, no cell; the block's size.
- `direction_codes.py`: the 26 directions as codes, their opposites and steps.
- `structure_arrays.py`, `structure_settings.py`, `size_law.py`: kind, length, the relation tables; the one size constant; length = constant over quanta for both kinds.
- `neighbour_of.py`: how a kernel reads a relation, always from the table.
- `grid.py`: `Grid`, the one state object: it holds the arrays and the host tables and reads and stores rows. It decides nothing.

## Must not get wrong

- **`Grid` stores, logic decides.** `set_neighbours`, `set_block`, `set_cell`, `new_ids`, `grow`, `free_slot` and the `send_*` uploads are how the one state object is read and written. They stay here because every axiom's logic needs them and logic may not import another axiom's logic. No rule of a step is in them: `lay_lattice`, `start_grid` and `relink` in `logic/` decide what the relations are.
- **A code is a label of a relation.** But `step_of` is used to give each relation its part of a cell's block (the mouths, axiom 4; the pull across a cell, axiom 5) and to cut a block at a tear (axiom 6): inside a cell the block has sides. See "Open" in the axiom's overlay; do not claim a cell has no shape inside.
- **The joint's list is the truth**: when a joint lists volume v at `code`, v lists the joint at `opposite(code)`. Whoever changes one side sets the other.
- **Row `cell` of a box array is that cell's block for good.** No box changes cell; a used id is never used again.
- **One size constant** for volumes and joints. An empty cell is endlessly long.
- Definitions import only definitions.
