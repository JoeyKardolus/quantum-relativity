# physics/engine/

The engine (axioms in `docs/bible.md#physicsquantum-relativity`): the cells are the
grid. A cell has only relations and a length; each cell id owns one block of 216
compute boxes. GPU (Warp).

## Must not get wrong

- **One test file per owner axiom** (`tests/test_axiom_<n>_*.py`), checked at every step
  of two real runs (`tests/axiom_run.py`) plus staged cases. Never loosen one.
- **Boxes are not space.** Row `cell` of `state.quanta` and `state.phase` is that cell's
  block for good. No kernel reads another cell's boxes; energy moves between cells as
  numbers (`pour`). An emptied id is never used again.
- **Size is the law, nothing else**: `length = float32(size_energy[kind] / quanta)`,
  set only in `cell_laws`, which runs first and last in a step. Two constants (volume,
  joint), fixed once by `new_universe`.
- **Relations**: a pristine cell stores nothing (neighbours and a face joint's way come
  from its lattice id); a changed cell keeps its own list (`adj_over`, `sides_over`),
  copied to the GPU by `relink`. Only `collapse.py` and `tear.py` write them. A **way**
  is the pair of volumes a joint lies between: energy crosses and volumes press only
  along ways; edge and corner joints have none.
- **The total is not fixed, the books are exact** (`totals`): amount = start + made -
  taken with joints - taken off volumes.
- **Squeeze** = pull across (G Qa Qb / distance²) over the joint's push (P (Q/length)²).
  Above 1 the joint makes minus quanta up to balance; its energy never falls. Squeezed
  and `collapse_thin` times the start's quanta: it collapses.
- **Collapse** removes the joint, pours the higher-id volume into the lower, takes the
  joint's quanta off it. **Tear** keeps the id, adds a joint and a volume at new ids;
  halves hold a third each, so they are longer still and tear again: the size law's
  own chain, not a bug to patch here.
- **Crossing** needs heat: `jiggle` runs after the fluid, or a start at phase 0 has
  none and everything crosses at once.
- **Pull**: leapfrog wave over neighbours, weighted by distance; one neighbour a step
  at most. The thin joint web is the short way round a volume.
- **Axiom 7** (`quantum*.py`): a moving energy is a record in `state.quantum`; only a
  scene splits it. Bell tests: not built.
- **Gone**: universes side by side, empty space and margins, the ladder. `state.torn_from`
  tells the viewer where a tear's cells came from.

## Measured (2026-11-17, RTX 5080, seed 1)

Defaults, 32 places (32,768 cells), 5,000 steps: no collapse, tear or lump; energy
evens out, size 70,048 to 69,646, shrinking ever slower; 350 to 570 steps/s. 1.0 million
cells 21 steps/s, 2.1 million 10.6. Uneven start (0.3), 16 places, pull x10: 825 tears,
513 collapses in 12 steps, one volume left. Uneven start, 8 places, defaults: a tear
chain, 64 volumes to 5,237 in 20 steps, size 1,147 to 57 million, seconds a step.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
