# physics/engine

Layer 2: `three_cells()` lays a volume, a fresh joint and a volume in a line. `step(store)` moves the fluid, lets it cross, reads the squeeze, lets the squeezed joint make quanta and presses its fluid. `one_cell()` is layer 1's; `calm_cells()` lays rules.

One directory per axiom. Axioms 1 to 6 hold code; 7 waits for its layer.

- A step runs only over cells that hold their own boxes. Nothing is copied to the host inside a step.
- All of it runs the same on the CPU and on the card. Tests use the CPU.
- The same scene runs the same course at 6, 12 and 24 boxes across, counted in turns of a calm swirl (tested).
