# physics/scenes/spin/

Scorecard scene 10. The winding measure lives in `physics/measure/winding/`.

**Known answer**: a whirl turns only in whole steps and never unwinds; spin does not
change by itself.

**Pass mark** (written 2026-11-17 before any run; restated 2026-11-18 for the rebuilt
engine, before any run on it): 16 places, natural start, every rule on, 2,000 steps.
Every watched cell, energy volumes and joints, reads its own one whole turn in every
layer across its swirl axis at the start and at every step.

## Must not get wrong

- **Nothing is placed any more.** In the rebuilt engine every cell is born one whole
  swirl (axiom 2), so the scene watches cells as the start made them. The old scene
  wound a whirl into a cell by hand; that and its double whirl (+2) are deleted.
- **A swirl lives inside its cell**: only the cell's own block is read. This is the
  one scene that reads boxes, through the engine's `block` and `physics.measure`.
- The turns a cell should read come from its `swirl_axis` and `swirl_sense`
  (`expected_turns`); the reading comes from the fluid's phases. Never compare a label
  with a label.

## Not done

- **Spin one half (the belt trick)** needs a whirl turned against its surroundings and
  read after one and after two turns. It is not built: `NOT_ASKED`, counted as a miss.
- The engine's own finding (axiom 2, "Not done"): nothing keeps a cell a swirl, and
  volumes unwind within about 20 steps. Expect a miss.
