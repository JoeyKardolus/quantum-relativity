"""Axiom 11: a too-big volume tears; only that volume splits, its middle plus quanta turn minus into the new joint, ABAB holds after.

Audit answers (owner, 2026-11-16): a tear turns the middle of the torn volume's plus
quanta into minus to make the new joint, the mirror of collapse; only the too-big volume
splits, never its neighbours. Bible rule 8: the trigger is size. Default universe (seed
1, Settings(), axiom_run.py): the energy volumes alive after every step are the start's
plus the splits minus the collapses so far (each split adds exactly one volume; the
run is printed with its split count). Staged trigger, Settings() (split_across unchanged):
new_universe(5, 10, seed=1, unevenness=0, margin=10), 8 energy volumes, with energy
volume V (lattice 2, 2, 2) left with 100 quanta per box, so it wants a side 10 times
longer and grows. Up to 300 steps until the first split; at that step:
  - the cell graph holds exactly one more energy volume (only V split)
  - no box that belonged to any cell but V before the step got the split mark (its
    neighbours unchanged)
  - the minus total rose by exactly the quanta the tear turned minus, and that is more
    than 0 (the tear's plus quanta became the new joint, amount kept)
  - ABAB: 0 touching box pairs of two volumes, 0 volume boxes beside empty space, the
    joints one web
"""
import numpy as np

from physics.engine import Settings, State, host_boxes, new_universe, stats, step, totals
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.axiom_contacts import axiom_contacts

V = (2 * 7 + 2) * 7 + 2


@needs_gpu
def test_the_default_run_splits_nothing_but_the_too_big():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    last = run["steps"][-1]
    start = int((run["kinds"] == 1).sum())
    wrong = [row["step"] for row in run["steps"] if row["volumes"] != start + row["splits"] - row["collapses"]]
    print(f"default run, 300 steps: splits {last['splits']}, torn {last['torn']}, collapses {last['collapses']}, "
          f"energy volumes {start} -> {last['volumes']}; steps where the count is off: {len(wrong)}")
    assert not wrong


@needs_gpu
def test_only_the_too_big_volume_splits_and_its_tear_becomes_the_joint():
    settings = Settings()
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=10))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == V] = 100
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=graph)
    split_at, across = None, []
    for count in range(1, 301):
        before_owner = host_boxes(state)["owner"]
        before_volumes = int((state.kind[0] == 1).sum())
        before = totals(state)
        torn_before = int(state.torn[0])
        across.append(float(np.cbrt((host_boxes(state)["owner"] == V).sum())))
        step(state, settings)
        if state.splits[0] > 0:
            split_at = count
            break
    if split_at is None:
        print(f"no split in 300 steps; V's size across (boxes) {across[0]:.2f} .. {max(across):.2f}, at the end "
              f"{across[-1]:.2f}; split_across {settings.split_across}")
        assert split_at is not None
    after = totals(state)
    volumes_after = int((state.kind[0] == 1).sum())
    marks = state.marks.numpy()
    others_marked = int((((marks & 8) == 8) & (before_owner != V)).sum())
    torn = int(state.torn[0]) - torn_before
    touching, opened = axiom_contacts(state)
    web = stats(state, settings)["web_pieces"]
    print(f"split at step {split_at}; energy volumes {before_volumes} -> {volumes_after}; split-marked boxes that "
          f"belonged to other cells {others_marked}; tear {torn} quanta, minus rose {after['minus'] - before['minus']}; "
          f"touching {touching}, open {opened}, web pieces {web}")
    assert volumes_after == before_volumes + 1
    assert others_marked == 0
    assert torn > 0 and after["minus"] - before["minus"] == torn
    assert touching == 0 and opened == 0 and web == 1
