# physics/engine/

The engine (issue #3, bible rule 4): per Planck box one signed int32 (plus = energy volume,
minus = joint, 0 = no space) and a float32 fluid phase. Universes run side by side; the
detail ladder (`physics/ladder`) sits beside the boxes once attached.

## Must not get wrong

- **The amount ignoring sign is exact.** Every move is an integer transfer of whole
  quanta; the sum of |quanta| holds to the quantum (`totals`).
- **The fluid is tied to the quanta**: sqrt(|quanta|) e^{i phase}, never stored;
  `pair_rotation.py` hands over whole quanta. No box drops below 1 quantum by flow.
- **Quanta change sign only in collapse.** Hops stay inside one sign and one cell (two
  volumes only once their shared joint is empty); the leak moves plus quanta between the
  end boxes of a plus-minus...-plus line, never into the joint. `marks`: 1 collapse, 2
  owner change.
- **The cells are a forced lattice** (owner, 2026-11-16): `state.owner` per box,
  extended-lattice ids (a ring of empty cells round the real ones). Kind and neighbours
  come from the index alone (`cell_kind.py`, `lattice_kinds.py`, `shared_joint.py`); the
  leak takes a line only when every minus box on it is its end volumes' shared joint's.
  Never read labels off the signs.
- **Corners move, boxes follow late.** `move_corners.py` gathers forces per corner from
  its 8 cells (no float atomics: alone equals batched) and puts back steps that turn a
  Kuhn tet inside out. `drain.py` fills no box past the cell's packing (quanta / corner
  volume) or the leaving box's own quanta, else waits: uncapped draining heaped joints
  to squeeze 2 to 6 and capped substeps at 64 (measured 2026-11-16).
- **No lattice** (`from_host` without `lattice`): nothing leaks or moves; single-rule
  test starts only. Starts built from `new_universe` pass `lattice=lattice_of(even)`.
- **Pull is a wave** (`pull_wave.py`); 0 in empty boxes and beyond walls.
- **Universes never touch.** Reads go through `inside`; alone equals batched (test).
- **The boxes are the truth; the ladder is beside them.** `ladder_sync` writes changed
  blocks after each step; nothing reads back (a COARSE read-back adds rounding noise).
- **Volume split** (new lattice planes) is not built.
- **No dark energy, no base push.** Never tune a default to pass a test.
- **Substeps resolve the steep push** (`substeps.py`, numerics): each universe cuts its
  fluid step so no same-sign face turns more than max_turn (0.25 rad) a substep.

## Measured (2026-11-16, RTX 5080, defaults)

Seed 1 default start (216^3 boxes, 27 volumes), 300 steps: 31 to 69 steps/s, 5 to 9
substeps, squeeze max 0.67 falling to 0.50, no collapse, no lumps, one web, no volume
unwrapped, drift 0, occupied x1.40 at 100 and x1.75 at 300. Straight volume-to-volume
lines through their own joint fall from 100% to 67 to 88% (5411/6128 at 25, 8662/11948
at 300); the leak takes only those (12851/12851 boxes at 300). `bench/CLAUDE.md` has sizes.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
