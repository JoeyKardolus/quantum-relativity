# Axiom 1: energy, definitions

What things are under this axiom. Each file's docstring is its contract.

- `energy_arrays.py`: `quanta` (one whole signed number a box), `cell_quanta`, the step's ledgers, the books' counts.
- `energy_settings.py`: `step_time` and the start's energies. There is no pressure setting.
- `pair_rotation.py`: the exact step of two waves trading fluid, inside a cell and (in `cross`, axiom 4) across a joint. There is no second law for a cell's side.
- `whole_move.py`, `whole_shares.py`, `cell_sum.py`: whole quanta, cut at 0, dealt exactly, counted exactly.
- `draw_uniform.py`, `passes.py`: the one source of random numbers (the start's uneven field too) and one pass number per use.
- `book_law.py`: what the totals must be.

## Must not get wrong

- **Whole numbers only.** A rounding inside a step is `floor(x + uniform)`, the uniform from `draw_uniform` with its own pass. The start alone rounds to the nearest whole number, once.
- **Two equal waves are left exactly as they are.** `pair_rotation` turns only what two boxes differ by. The old step also turned what they share; a box at the block's side, with fewer partners, then fell out of step, and the block's shape pushed the fluid about like a force. Do not bring that turn back.
- **No law may invent the fluid beyond a side.** The old `outflow_law` stepped a mouth box against a made-up wave; it turned the box even when nothing left, and that pumped motion out of nothing (one moved quantum grew to half of all energy changing box a step). What leaves a cell is one `pair_rotation` with the real box that faces it.
- **The sign.** Fluid flows from the wave whose phase is ahead. `cross` (axiom 4), `swirl_phase` (axiom 2) and `pull_fluid` (axiom 5) all rest on that; change one and all must change.
- **`book_law` returns four numbers.** `taken` ties what collapses took off volumes to what their joints held; a count that only "rises" proves nothing.
- **No phase is ever drawn.** There is no random angle in the engine.
- **The same four numbers give the same draw** on any device.
- Definitions import only definitions, of axioms 3 and 2 and this one.
