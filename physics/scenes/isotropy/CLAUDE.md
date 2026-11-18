# physics/scenes/isotropy/

Scorecard scene 9.

**Known answer**: space has no preferred direction: at the same straight-line distance
the pull and the flow are the same in every direction.

**Pass mark** (written 2026-11-17 before any run): 48 places, one volume packed 4
times in an even grid, 2,000 steps, every rule on. For every pair of places at the same
straight-line distance, one on an axis and one off it, pull off-axis over on-axis is
within 10% of the box law's ratio and flow off-axis over on-axis within 10% of 1.

## Must not get wrong

- **The box itself is a cube.** Even a law with no preferred direction is not the same
  on and off an axis in a wrapping cube (about 0.94 at 3 volumes in an 8-volume box).
  The pull is read beside `box_field`, never beside 1.
- **What the lump adds is the difference from the even twin**, same seed, stepped side
  by side. Every mirror and swap of a place is averaged.
- The pull is the mean over the steps after `settle`: it is a wave that keeps swinging.

## Changed with the rebuilt engine

- The old known-answer run had crossing off, so the lump kept its energy and its pull
  was one point's pull. There is no off switch any more. The reading is now proved on
  the law itself (`definitions/tests`) and on step 0 of a real start (`logic/tests`).
- A pull 3 volumes out is 6 cells away; at one neighbour a step a trace arrives after 6
  steps and the bulk far later (axiom 5, "Not done"). A short run reads no pull there.
