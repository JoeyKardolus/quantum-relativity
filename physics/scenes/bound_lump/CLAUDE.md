# physics/scenes/bound_lump/

Scorecard scene 2.

**Known answer**: a bound thing keeps its size while space grows.

**Pass mark** (written 2026-11-17 before any run): at the end the fullest volume is a
lump (2 times an even start volume or more); over the second half of the run its length
changes by less than 10% while the other volumes' mean length grows by more than 5%;
the universe stands at 5,000 and at 20,000 steps.

## Must not get wrong

- **A cell's size is its energy and nothing else** (axiom 3). So "keeps its size" means
  "keeps its energy": the lump's length is the size constant over its quanta. The scene
  reads the engine's `length`; it never computes a size another way.
- The lump is picked at the end among volumes that lived since the start; a cell id is
  the same cell for good, so its earlier lengths are its own.
- Other volumes holding 0 quanta are endlessly long and are left out of the rest's mean.

## Not done

- One volume is a small "bound thing". A lump over several volumes (the structure
  scene's joined region) is not followed here.
