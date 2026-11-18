# physics/view/colours/

Colours: energy volumes red, joints blue. A cell's fill is clear at the calm
background's packing and turns solid as it packs, so a lump stands like a mass in
empty-looking space; its outline shows the pattern either way.

- `definitions/`: the two hues, the scale's fixed numbers, the opacity law, the
  outline's opacity.
- `logic/`: takes the start's marks, reads each packing against them, makes RGBA, and
  picks the cells the whole grid shows.
- `tests/`: from raw packings to drawn colours, with the engine's own proportions.

## Must not get wrong

- **The hue is the kind and never changes.** Only the opacity carries the packing.
- **One fixed scale.** A fill's opacity is 0 at the start's mean packing or below and a
  straight line up to 1 at `SOLID` (4) times it. It is never ranked or stretched to a
  frame, so an opacity means the same packing in every frame of a run.
- **The whole grid raises the opacity to `WHOLE_POWER`** (3.8) when it draws, because
  the eye looks through many cells, and shows only `packed_cells`.
- **The outline is not the packing.** Its opacity follows only the cell's size on
  screen (`outline_opacity`).
- No number here is a physics setting and none is tuned to make a run look good.

## Assumed

- **Each kind is read against its own start mean.** By the engine's one size law a
  joint holds less energy than a volume, so it is far less packed. Read against one
  mark, every joint would be clear for ever. The owner may want one mark instead.
- In a scene that starts with lumps the mean is above the background, so the
  background is clear from the first frame.
