# physics/engine

Layer 2: `three_cells()` lays a volume, a joint and a volume in a line; `step(store)` moves the fluid inside each, lets it cross the joint and presses the joint. `one_cell()` is layer 1's; `calm_cells()` lays cells that are still only a rule.

One directory per axiom. Axioms 1 to 5 hold code; 6 and 7 wait for their layer.

- A step runs only over cells that hold their own boxes. Nothing is copied to the host inside a step.
- All of it runs the same on the CPU and on the card. Tests use the CPU.
- A new layer adds files. It does not rewrite the ones below.
