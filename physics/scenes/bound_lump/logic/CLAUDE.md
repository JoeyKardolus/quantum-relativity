# physics/scenes/bound_lump/logic/

`bound_lump` keeps every start cell's length per sample, then picks the lump at the end.

- Only start ids are kept (`side ** 3`): a volume a tear made has no start length.
- `rest_length` is the mean over the other start volumes living and holding energy at
  that sample; NaN when there is none.
- If no start volume lives at the end the scene raises: there is no lump to follow.
