# physics/scenes/orbit/logic/

- `orbit_start` places quanta with `packed_start`, then writes the ramp into the small
  volume's `phase` block and nothing else. Quanta, sizes and books are untouched by the
  ramp: a phase is not energy.
- The small volume lies along the first axis from the big one; the ramp runs along the
  second, so the turning is read in the plane of those two (`went_round`).
- **The twin is subtracted at every sample.** It is the same seed with the big volume
  only, stepped to the same step. In the rebuilt engine an even grid moves energy by
  itself (every cell is a swirl), and without the twin that motion swamps the blob
  within three steps. At step 0 the blob sits exactly `distance` volumes out; the test
  holds this.
