# physics/scenes/bound_lump/definitions/

- `second_half_change` is the last sample over the sample at the middle index
  (`len // 2`): the mark is judged on the second half so the start's own settling does
  not count for or against the lump.
- 10% and 5% were written before any run. What a lump is comes from
  `volumes/definitions/lump_factor.py`.
