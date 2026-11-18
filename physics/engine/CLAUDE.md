# physics/engine

Layer 1: one energy volume. `one_cell()` lays one whole swirl in its boxes; `step(store)` moves the fluid one step; `calm_cells()` lays cells that are still only a rule.

One directory per axiom. Axioms 1 to 3 hold code; 4 to 7 wait for their layer.

- A step runs only over cells that hold their own boxes. Nothing is copied to the host inside a step.
- All of it runs the same on the CPU and on the card. Tests use the CPU.
- A new layer adds files. It does not rewrite the ones below.
