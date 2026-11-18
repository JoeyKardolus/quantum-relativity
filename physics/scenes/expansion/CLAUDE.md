# physics/scenes/expansion/

Scorecard scene 6, expansion shape, and one of the two the fit may see.

**Known answer**: the real universe grew, first slowing, then speeding up.

**Pass mark** (written 2026-11-17 before any run, `definitions/pass_mark.py`): size at
the end 1.05 times the start's or more, the growth rate's trend below 0 over the first
half and above 0 over the second, and the universe still standing at 5,000 and at
20,000 steps (24 places a side). Whether it stands is the fit's end rule, judged
outside this package and handed in.

## Must not get wrong

- **Size is the sum of the cells' lengths** (`measure_grid`), nothing else. The universe
  wraps: there is no edge to measure at.
- A cell holding 0 quanta is endlessly long; the engine leaves it out of the sum and
  counts it (`endless`). A size that jumps down may be a volume that just emptied.
- The scene returns numbers; it never judges itself.

## Not done

- Nothing has been run at the scorecard's size on the rebuilt engine. At 8 a side the
  default universe falls into one volume within 30 steps.
