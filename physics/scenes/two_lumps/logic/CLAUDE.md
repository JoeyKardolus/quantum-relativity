# physics/scenes/two_lumps/logic/

- `pair_run` steps the pair and the lone lump in the same loop, so both are read at the
  same steps. The tilt is the mean over the steps after `settle`: the pull is a wave
  that keeps swinging.
- The lumps lie along the first axis, the first one at or below the middle so the pair
  never wraps.
- The tube for the closing reading is 5 volumes wide and reaches `apart // 2` volumes
  either side of a lump, so the two tubes never share a volume.
- `two_lumps` refuses a distance under 2 volumes or over a third of the volumes a side:
  nearer, the lumps touch; farther, a lump is as near its neighbour's copy.
