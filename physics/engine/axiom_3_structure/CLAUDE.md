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

- One size constant: the start's joints are 300 long, its volumes 10.
- Wrapped means every neighbour of a volume is a joint and it has one. A side is a relation, so no count asks for 26 sides.
- When a joint goes, the joints on its opposite sides touch (`close_web`, axiom 6); the web stays one piece.

## Real runs (16 a side, seed 1)

- Every judged pattern count is 0 and the web is one piece at every step of all four runs: through 1,928 collapses (default) and 3,024 tears (uneven).
- Default run ends as 4 volumes and 1,656 joints, 1,590 of them `inner_joints`.

## Not done

- **After a tear two pieces lie on one side of the joints beside them**, doubling every step (`crowded_mouths`).
- **Lumps hardly warp the grid**: volumes are 0.48% of its length and no joint changes length while it holds. No crossing reads a length, and a settling (axiom 7) reads none.
- **The default run ends with no A B A.** Nearly every joint left lies inside one merged volume (`inner_joints`, measured, not judged).
- **"Wrapped" is not judged by a count.** `bare` asks for one living joint; a volume stripped to 6 of its 26 joints passes. After tears most volumes have fewer than 26.

## Open

- **A cell has a shape inside.** A direction of the fluid has to mean a neighbour, so each relation gets the boxes of the block that look its way: 16 for a face, 4 for an edge, 1 for a corner (`mouth.py`); a swirl has an axis in it and a tear cuts it in layers. Two mouths that face each other trade box for box (axiom 4), so a relation carries by its 16, 4 or 1 pairs.
