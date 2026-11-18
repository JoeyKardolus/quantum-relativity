# physics/view/colours/

Colours: energy volumes red, joints blue, and how solid a cell looks follows how packed
it is, so a packed lump stands in the thin rest like a solid mass.

- `definitions/`: the two hues, the scale's fixed numbers, the opacity law.
- `logic/`: takes the start's marks, reads each packing against them, makes RGBA.
- `tests/`: from raw packings to drawn colours, with the engine's own proportions.

## Must not get wrong

- **The hue is the kind and never changes.** Only the opacity carries the packing.
- **One fixed scale.** Opacity is packing over the start's mean, divided by `SOLID`
  (4), cut off at 1. It is never ranked or stretched to a frame, so an opacity means
  the same packing in every frame of a run.
- **The whole grid raises the opacity to `WHOLE_POWER`** (3.8) when it draws, because
  the eye looks through many cells: the start's mean becomes 0.005, four times it solid.
- No number here is a physics setting and none is tuned to make a run look good.

## Assumed

- **Each kind is read against its own start mean.** By the engine's one size law a
  joint holds less energy than a volume, so it is about 30 times longer and about a
  million times less packed (measured, defaults, seed 1: 8e-4 against 709). Read against
  one mark, every joint would be clear for ever. The owner may want one mark instead.
