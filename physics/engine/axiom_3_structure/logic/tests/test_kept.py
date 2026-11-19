"""A store written to a file and read back runs on exactly as the one that was never written."""
import numpy as np

from physics.engine import step, three_cells
from physics.engine.axiom_3_structure.logic.kept import keep, kept


def test_a_kept_store_read_back_runs_the_same_course(tmp_path):
    store = three_cells()
    for _ in range(5):
        step(store)
    keep(store, tmp_path / "bench.npz")
    back = kept(tmp_path / "bench.npz", three_cells())
    assert back.step == 5 and back.count == 3 and back.side == 3
    for _ in range(20):
        step(store)
        step(back)
    assert (store.box_quanta.numpy() == back.box_quanta.numpy()).all()
    assert (store.cell_flow.numpy() == back.cell_flow.numpy()).all() and np.array_equal(
        store.box_phase.numpy(), back.box_phase.numpy())
