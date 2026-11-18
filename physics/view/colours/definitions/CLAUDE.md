# physics/view/colours/definitions/

What a colour is here. Constants and one formula.

- `palette.py`: `VOLUME_RED`, `JOINT_BLUE`, `SOLID`, `WHOLE_POWER`.
- `opacity_law.py`: opacity = packing over the start's mean, over `SOLID`, between 0
  and 1. A straight line through 0: an emptied cell is fully clear.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine.
- Changing `SOLID` or `WHOLE_POWER` changes what every picture means. `tests/` pins the
  numbers they give: the start's mean is 0.25 in a slice and 0.005 in the whole grid;
  32 places of even cells still let through most of what lies behind them.
