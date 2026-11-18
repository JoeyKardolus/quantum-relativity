# The score

One number between 0 and 1 for a universe's run:

    score = hold x (E + S) / 2

- `E`, expansion shape: did the size, and the gap between lumps, grow steadily, slowing
  first and then speeding up.
- `S`, structure: four gains since the start (region contrast, lump share, lumps that
  hold, the spread of lump sizes).
- `hold`: the share of the start's volumes still alive times the share of the volumes'
  size not overstretched.

`definitions/` holds the scales and every formula; `logic/score_samples.py` puts them
together; `tests/` scores a real universe.

## Must not get wrong

- **Nothing happening scores exactly 0.** Every part is a change since the start.
- **Fallen into one cell, or torn without end, scores about 0**, through `hold`.
- **No cliff.** `hold` falls in a straight line.
- **tanh caps every part**, so a runaway earns no more than steady growth.
- **Change the score only before a search**, and write down what the old one let through.
- **The score reads sample dicts by name**; it imports nothing from `measures` or `search`.

## Changed for the rebuilt engine (2026-11-18, before any search on it)

- The sealing part (`Ss`) is gone: the engine has no seal. Structure is now the mean of
  four parts, not five.
- `spent_share` is a share of the volumes' size (see `measures/CLAUDE.md`).
