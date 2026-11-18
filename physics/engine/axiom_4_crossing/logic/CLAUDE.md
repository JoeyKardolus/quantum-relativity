# Axiom 4, logic

`cross.py`: one launch over the own slots; only a joint (odd place) with both volumes own does anything.

- A joint writes only its two volumes' facing boxes, so no box is in two pairs of the launch.
- The cell totals and the heat are added with `atomic_add`: whole numbers, so the order does not matter.
- Measured: how fast the two volumes even out depends on the boxes across (750 steps at 6, still sloshing after 9,600 at 24). That is wrong: boxes are not space.
- A calm volume beside an own joint is not crossed into yet (layer 3).
