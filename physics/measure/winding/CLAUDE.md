# Measure: winding

How many whole turns a cell's fluid makes round an axis of its own block (axiom 2:
"cells are swirls ... in whole steps"). Going once round a closed loop of boxes, the
phase must come back to itself, so it has turned a whole number of times: 0, 1, -1, 2.
That number is the winding.

`definitions/` says how a loop is walked, what a phase step is, and what a cell that is
one whole swirl reads. `logic/` reads the winding of every smallest loop, and a cell's
turns layer by layer. `tests/` checks the whole measure on fields with known swirls.

## Must not get wrong

- **A swirl lives inside its cell.** The measure reads one block (`state.block("quanta",
  cell)`, `state.block("phase", cell)`). It never joins two cells' boxes: boxes are not space.
- **Whole by construction.** A winding is always a whole number, so "is it whole" is never
  the test. The test is whether it equals `expected_turns` for the cell's own
  `swirl_axis` and `swirl_sense`.
- **Signs.** Loops are walked right-handed about the axis. The engine winds by minus its
  sense, and for axis 1 takes its two coordinates in the other order; `expected_turns`
  holds that table, and the test on the real start pins it.
- **An empty box has no phase.** Every smallest loop touching it counts 0, so a layer can
  read 0 because a box emptied, not because the swirl unwound.

## Measured (16 a side, seed 1; cells not reading their expected turns in all six layers)

- Start: 0 of 4,096.
- Held run, volumes of 512: 0 at step 1, 487 at step 5, 507 at step 9, 512 at step 21.
- Held run, joints of 3,584: 0, 0, 875, 70 at the same steps; 3,277 at step 8 and 1,761
  at step 24. The count goes up and down.

## Not done

- **This is stricter than the engine's own check.** Axiom 2's `check_step` reads the
  middle ring and asks four layers of six; `cell_turns` adds every smallest loop and the
  scene must ask all six. The two counts differ; neither is tested against the other.
- **Why the joints' count goes up and down is not found** (an emptied box, or a step of
  exactly half a turn). Until it is, a joint's reading is not to be trusted.
- **Nothing here reads "oscillate".** The winding says whether a cell is still wound, not
  whether its fluid swings.
