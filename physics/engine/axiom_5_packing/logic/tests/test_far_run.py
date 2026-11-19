"""A far run logs a line every so many turns, keeps its state, and a kept state runs on as the run itself would."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_5_packing.logic.far import NEAR, far, resumed


def test_a_far_run_logs_keeps_and_a_kept_state_steps_on_box_for_box(tmp_path):
    store = far(tmp_path, minutes=5, turns=6, every=2, keep_every=4)
    lines = (tmp_path / "far.log").read_text().splitlines()
    assert [line.split()[1] for line in lines if line.startswith("turn")] == ["0", "2", "4", "6", "6"]  # the last: as it ended
    assert "kept at turn 4" in lines[3] and lines[-1].startswith(f"ended at step {store.step}")
    assert store.near == NEAR and store.count == 81  # the first layer stands: 27 volumes, 54 joints
    roomy = resumed(tmp_path / "last.npz", more=2)  # the same universe with room for two layers more
    assert roomy.side == store.side + 8 and roomy.count == store.count and roomy.step == store.step
    again = resumed(tmp_path / "last.npz")
    for _ in range(30):
        step(store)
        step(again)
    assert again.near == NEAR and again.step == store.step
    for name in ("cell_quanta", "cell_flow", "cell_pull", "box_quanta", "box_phase"):
        assert np.array_equal(getattr(store, name).numpy(), getattr(again, name).numpy()), name
