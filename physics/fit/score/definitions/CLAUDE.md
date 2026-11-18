# The score, definitions

The scales and the formula of every part. Each file's docstring is its contract.

- `score_marks.py`: where each part stops counting more, and the fewest samples.
- `gain.py`: tanh(change / scale), 0 for a loss.
- `length_shape.py`: grew x steady x (1 + dip) / 2 for one length.
- `structure_parts.py`: the four structure gains.
- `hold_factor.py`: alive x (1 - spent share).

## Must not get wrong

- **The scales are not fitted.** They say only where a part saturates.
- **The dip is in the speed of growing**, cut in three thirds of the observer's time,
  not in the rate ln a (which in the real universe only ever falls).
- **The spread of lump sizes counts only as far as energy gathered into lumps.** Lumps
  that only melt leave a few lying further apart in size; that must score nothing.
- Pure functions; nothing here imports the engine or logic.
