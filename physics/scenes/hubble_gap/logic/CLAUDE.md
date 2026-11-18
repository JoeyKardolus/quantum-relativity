# physics/scenes/hubble_gap/logic/

`hubble_gap` runs `expansion` once and reads `size_rate`.

- It returns the whole expansion result under `expansion`, so the scorecard judges
  both scenes from one run.
- `late_over_early` is NaN when the early rate is exactly 0; nothing is nudged to avoid it.
