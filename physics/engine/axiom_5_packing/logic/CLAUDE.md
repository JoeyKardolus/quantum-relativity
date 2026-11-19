# Axiom 5, logic

`between.py`: the way between volumes through the cells' sizes. `pull.py`: all pairs on the host; writes `cell_pull`, adds to `cell_flow`. `squeeze.py`, `press.py`: the joints. `lumps.py`, `staging.py`: staging. `ball.py`: the grown ball, kept in tmp, and the window's scenes. `weighed.py`: a staged ball against the plain one. `line.py`: benches, tests only.

- Order in a step: `pull`, `squeeze`, then `make` and `press`.
- `cell_past` lives on the host.
- A volume must never enter its own sum.
- A collapsed joint keeps its distance and is not squeezed.
- `lumps.py` imports the engine's door; the door must not import it.
- Tests share the kept ball (2,705 steps, grown once).
