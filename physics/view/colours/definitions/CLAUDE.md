# physics/view/colours/definitions/

What a colour is here. Constants and one formula.

- `palette.py`: `VOLUME_RED`, `JOINT_BLUE`, `SOLID`, `WHOLE_POWER`, and the outlines'
  `OUTLINE`, `OUTLINE_FULL_AT`, `OUTLINE_INSET`.
- `opacity_law.py`: a fill's opacity: clear at the start's mean packing or below, a
  straight line up to solid at `SOLID` times it.
- `outline_opacity.py`: the outlines' opacity from a place's width on screen: full
  from `OUTLINE_FULL_AT` pixels up, fading in proportion below.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine.
- Changing `SOLID` or `WHOLE_POWER` changes what every picture means. `tests/` pins the
  numbers they give: the start's mean is clear, 2.5 times it is 0.5 in a slice and 0.07
  in the whole grid.
