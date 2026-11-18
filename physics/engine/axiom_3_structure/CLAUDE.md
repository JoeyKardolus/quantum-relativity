# Axiom 3: structure

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 3):

> Structure: plus never touches plus; minus wraps every plus and forms one web. The cells are the grid: a perfect A B A B pattern that never breaks and never flickers; a cell has no geometry of its own, only relations to its neighbours and a size set by its energy alone, more energy smaller, for volumes and joints alike. Lumps warp the grid by making their cells small. Planck boxes are not space: they sit inside each cell only to compute its fluid.

They win over this file and the code. `tests/check_step.py` checks the axiom on real runs.

## Clause by clause

- "never touches", "wraps", "one web", "never breaks, never flickers": `logic/check_pattern.py` counts `touching`, `bare`, `lone_joints`, `web_pieces`, `flickered` and the table faults.
- "only relations": the tables (`definitions/grid.py`, `logic/relink.py`).
- "a size set by its energy alone": `size_law.py`, `logic/size_cells.py`, one constant.
- "lumps warp the grid": `logic/measure_grid.py`, measured.
- "boxes are not space": a block is its cell's for good; what a mouth sends is dealt evenly over the facing mouth, never box to box (`cross`, axiom 4).

## Assumed

- One size constant. A joint's length follows from the balance the start is laid in (`lay_start`): at the defaults 22 against a volume's 10.
- Wrapped means every neighbour of a volume is a joint and it has one. A side is a relation, so no count asks for 26 sides.
- When a joint goes, the joints on its opposite sides touch (`close_web`, axiom 6); the web stays one piece.

## Real runs (16 a side, seed 1)

- Every judged pattern count is 0 and the web is one piece at every step of all five runs: through 1,874 collapses (collapsing) and the uneven run's tears. The default run has no event.

## Not done

- **After a tear two pieces lie on one side of the joints beside them**, doubling every step (`crowded_mouths`).
- **Thin joints need most of the energy.** One size law makes a thinner joint a fuller one: default joints are 2.2 times a volume's length and hold 76% of all energy. For the owner.
- **No lump grows in 40 steps of the default run** (one expected failure). No crossing reads a length.
- **The collapsing run ends with no A B A**: joints inside merged volumes (`inner_joints`, measured, not judged).
- **"Wrapped" is not judged by a count**: `bare` asks for one living joint.

## Open

- **A cell has a shape inside.** A direction of the fluid has to mean a neighbour, so each relation gets the boxes of the block that look its way: 16 for a face, 4 for an edge, 1 for a corner (`mouth.py`); a swirl has an axis in it and a tear cuts it in layers. Two mouths that face each other trade box for box (axiom 4), so a relation carries by its 16, 4 or 1 pairs.
