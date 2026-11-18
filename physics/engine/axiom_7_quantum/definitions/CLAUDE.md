# Axiom 7: quantum, definitions

What things are under this axiom. Each file's docstring is its contract.

- `phase_arrays.py`: `phase`, float64 per box; the host store of open splits.
- `wrap_phase.py`, `joined_phase.py`: a phase is an angle; two fluids in one box join by their quanta.
- `quantum_settings.py`: the range a split's speed is drawn from.
- `split_store.py`: the open splits and their halves as tables of whole numbers; this step's counts and settlings.
- `speed_draw.py`, `pointer_law.py`: the speed, log-uniform in the split's own draw; the pointer's cell by share.

## Must not get wrong

- **The store holds whole numbers only.** No speed, no turn count: `turn_splits` works the speed again from the split's own draw and drops it. Adding a float column breaks 7e, and `check_step` counts it (`fractions`).
- **A split is a claim, not a container.** The quanta stay in the boxes; a row says how many of which cell belong to which split. No quantum is claimed twice: a half goes on with its cell's fluid (`carry_on`), and only outflow that is no open split's opens a new one. `check_step` counts `over_claimed`.
- **The tables stay in rising order** (`source`; then `part_source`, `part_cell`): `pointer_law`, `amount_of` and every `searchsorted` rest on it. Only `set_splits` (logic) writes them whole.
- **A split's quanta are the sum of its halves** (`amount_of`); there is no second number to keep in step.
- **Phases are float64**, wrapped at every write.
- Definitions import only definitions.
