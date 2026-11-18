# View, definitions

`camera.py`: where the eye is, and the wheel's range. `shaders.py`: the two programs the card runs to draw a streak, and how solid it is.

- The cell is placed as a cube one unit wide, whatever its energy; distances are in cell edges. The cube is never drawn.
- The first view looks nearly along the swirl's axis, so the open core shows as a hole.
- The red is one colour, laid over black by how solid it is, so many streaks over each other go to solid red and never to yellow or white.
- `FULL` and `THICK` only draw. Set by eye once; do not move them.
