"""The same start gives the same run, box for box: nothing in the engine is drawn by chance."""
from pathlib import Path

import numpy as np

from physics.engine import one_cell, step, three_cells
from physics.engine.axiom_6_limits.logic.tests.test_grow import _grown


def _after(steps: int, lay=one_cell) -> tuple[np.ndarray, np.ndarray]:
    store = lay()
    for _ in range(steps):
        step(store)
    return store.box_quanta.numpy(), store.box_phase.numpy()


def test_the_same_start_gives_the_same_run():
    first, again = _after(200), _after(200)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], _after(0)[0])  # and the fluid did move


def test_the_same_start_gives_the_same_crossing():
    first, again = (_after(200, three_cells) for _ in range(2))
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])


def test_the_same_start_gives_the_same_growth():
    first, again = (_grown(layers=2)[0] for _ in range(2))
    assert first.step == again.step and first.count == again.count > 27
    assert np.array_equal(first.box_quanta.numpy(), again.box_quanta.numpy())
    assert np.array_equal(first.box_phase.numpy(), again.box_phase.numpy())
    assert np.array_equal(first.cell_quanta.numpy(), again.cell_quanta.numpy())


def test_no_engine_file_names_a_seed_a_draw_or_chance():
    engine = Path(__file__).parents[1]  # physics/engine
    words = ("se" + "ed", "draw_" + "uniform", "ran" + "dom")
    named = [f"{file.relative_to(engine)}: {word}" for file in engine.rglob("*.py")
             if "tests" not in file.parts for word in words if word in file.read_text().lower()]
    assert named == []
