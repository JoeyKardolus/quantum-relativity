# physics/engine/

The engine (issue #3, bible rule 4): per Planck box one signed int32 (plus = energy volume,
minus = joint, 0 = no space) and a float32 fluid phase. Universes run side by side; the
detail ladder (`physics/ladder`) sits beside the boxes once attached.

## Must not get wrong

- **The amount ignoring sign is exact.** Every move is an integer transfer of whole
  quanta; the sum of |quanta| holds to the quantum (`totals`).
- **The fluid is tied to the quanta**: sqrt(|quanta|) e^{i phase}, never stored;
  `pair_rotation.py` hands over whole quanta. No box drops below 1 quantum by flow.
- **Quanta change sign only in collapse (minus to plus) and a split's tear (plus to
  minus; owner question on issue #1).** Hops stay inside one sign and one cell; the
  leak moves plus quanta between the end boxes of a plus-minus...-plus line, never into
  the joint. `marks`: 1 collapse, 2 owner change, 8 split.
- **The cells are a forced lattice** (owner, 2026-11-16): `state.owner` per box, ids
  over the batch's `extent` per axis; each universe has its own `cells` (nx, ny, nz).
  Kind and neighbours come from the index alone (`cell_kind.py`, `lattice_kinds.py`,
  `shared_joint.py`); never read labels off the signs. `host_boxes` renumbers owners to
  the universe's own lattice.
- **Corners move, boxes follow late.** `move_corners.py` gathers forces per corner from
  its 8 cells (no float atomics) and puts back steps that turn a Kuhn tet inside out or
  press a joint thinner than 1 box across (owner, 2026-11-16: a joint box always lies
  between two volumes, 26-neighbour; test). `drain.py` fills no box past the cell's
  packing, else waits.
- **Volume split** (`volume_split.py`): a volume whose corner volume reaches
  split_across**3 tears across its longest axis; two whole lattice planes go in, so its
  whole layer splits; the middle part's plus boxes turn minus (the tear). A split that
  leaves a part without quanta or inside out is put back.
- **No lattice** (`from_host` without `lattice`): nothing leaks or moves; single-rule
  test starts only. Starts built from `new_universe` pass `lattice=lattice_of(even)`.
- **Pull is a wave** (`pull_wave.py`); 0 in empty boxes and beyond walls.
- **Universes never touch.** Reads go through `inside`; alone equals batched (test).
- **The boxes are the truth; the ladder is beside them.**
- **No dark energy, no base push.** Never tune a default to pass a test.
- **Substeps resolve the steep push** (`substeps.py`, numerics).

## Measured (2026-11-16, RTX 5080, defaults)

Seed 1 default start (216^3, 27 volumes), 300 steps: ~30 steps/s, no split (volumes
~11 across; split_across 20), no collapse, no lumps, one web, drift 0, no touching
volume boxes, occupied x1.30 at 50, x1.75 at 300, gains per 50 steps 102k, 36k, 36k,
32k, 24k, 29k (slows). Viewer start (8^3 cells of 6^3): no split; thinnest joint 0.12
boxes before the joint guard, 1.09 after. Split test: halves 1000 -> 2227 boxes in 75
steps.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics/engine     # skips without a CUDA GPU
```
