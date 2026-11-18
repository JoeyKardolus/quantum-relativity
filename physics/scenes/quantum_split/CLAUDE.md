# physics/scenes/quantum_split/

Scorecard scene 11.

**Known answer**: a split energy is found whole in one place, in each place as often
as that place's share of it (a 50/50 split in half the draws, a 70/30 split in 70%).

**Pass mark** (rewritten 2026-11-18 for the rebuilt engine, before any run of this
scene): 32 places, natural start, 20 steps, then a detector on every open split. At
least 200 splits settle, every one whole, and the share that settles in its largest
half's cell is within 0.05 of the mean share those halves held.

## Must not get wrong

- **Nothing is staged any more.** The old scene named a packet and handed halves from
  cell to cell by hand (`quantum_packet`, `quantum_hand`), because the old crossing
  took all direction off energy. Those doors are gone: outflow that meets a fork splits
  by itself. The scene lets the engine split, then only reacts (`react`, the engine's
  public detector) and reads the store.
- **The shares are read one step before the settling.** A detector acts at the end of
  the next step, after the halves have gone on with their fluid for that step, so the
  expected share is a step old. Read the two numbers with that in mind.
- Only the engine's own record of a settling (`quantum["settled"]`) is read; the scene
  never decides where a split landed.

## Not done (each counts as a miss, `NOT_ASKED`)

- Two routes brought back together in step and out of step: in the engine phase plays
  no part in where a split settles (axiom 7, "Not done"), and nothing can route halves.
- No message: a far side's odds with and without a near reaction. It needs two
  detectors at chosen times on one split's two halves.
- Bell test (filter angles on pairs of whirls): not built.
