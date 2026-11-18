# Axiom 4: crossing, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `cross.py`: for every living way, each pair of boxes that face each other across the joint takes the fluid's own step (`pair_rotation`, axiom 1); a share of what crosses loses its direction (friction), and that is booked as heat.

Four kernels a pass, one job each: `_trade` (the pairs' step), `_rub` (friction's share), `_move` (the boxes give and take), `_book` (the ledgers and the heat).

## Must not get wrong

- **The crossing is the fluid's own step and nothing more.** Fluid flows from the box whose phase is ahead; a fuller box evens out with an emptier one; two equal waves are left exactly as they are. Nothing here drives the fluid: an even universe of one phase is bit for bit the same after any number of crossings, at any friction. A test pins it.
- **Phases of two cells are read as they stand.** The energy is one fluid with one phase field. So every rule that turns a phase must turn it the same way on both sides of a joint: `pull_fluid` (axiom 5) turns a cell's whole fluid by its own pull, not only by the slope inside the cell.
- **Only through minus.** A way acts only when its joint is a living joint that holds minus energy and both ends are energy volumes. No box of a joint is written.
- **No mouth is in two ways of one pass** (`way_pass`, axiom 3); a mouth that several ways share splits its local time between them.
- **Books.** `way_out_a`, `way_out_b` are gross (some pairs of one way go one way, some the other); `way_moved` is the net. Heat leaves with the quanta that leave and never exceeds what a cell holds.
- **No total changes here.**

## Tests

`tests/conftest.py`: `universe` (a real 4-a-side start on the CPU), `still` (one phase everywhere), `ramp` (one volume set moving toward +x), `face_way`, `held_by`.
