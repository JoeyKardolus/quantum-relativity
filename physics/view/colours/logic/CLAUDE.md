# physics/view/colours/logic/

What happens when cells are coloured:

1. `start_marks.py`: once, at the start: the mean packing of the energy volumes and of
   the joints.
2. `relative_packing.py`: every frame: each cell's packing over its own kind's mark.
3. `cell_colours.py`: RGBA per cell: hue by kind, alpha by `opacity_law`.
4. `packed_cells.py`: which cells the whole grid shows at all: those whose fill is at
   least one step of 255 on its scale (about 1.7 times the mark).

## Must not get wrong

- **Marks are taken once per start**, never per frame. A reset takes new ones.
- `start_marks` raises when a kind has no cell or no packing: there is then nothing to
  read against, and a made-up mark would be a made-up picture.
- An id that is no cell has relative packing 0.
- `tests/` checks known alphas: relative packings 0.05, 1, 1.3, 2, 3, 4 give
  0, 0, 26, 85, 170, 255.
