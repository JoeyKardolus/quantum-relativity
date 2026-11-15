"""Space is filled with no gaps and no overlaps: every cell is closed and the right way out, and the cells' volumes
sum to the volume inside the outer surface, at the start and through collapses and splits (rule 1)."""
import numpy as np

from physics.cells import EngineSettings, cell_sizes, new_engine, step


def _hull(engine):
    """The volume inside the outer triangles."""
    outer = engine.faces[engine.face_cells[:, 1] < 0]
    first, second, third = (engine.corners[outer[:, column]] for column in range(3))
    return float(np.einsum("ij,ij->i", first, np.cross(second, third)).sum() / 6)


def _open_edges(engine):
    """Triangle sides of a cell's surface that no other triangle of the same cell turns back along."""
    inner = engine.face_cells[:, 1] >= 0
    cell = np.concatenate([engine.face_cells[:, 0], engine.face_cells[inner, 1]])
    ring = np.concatenate([engine.faces, engine.faces[inner][:, ::-1]])
    sides = {(int(owner), int(ring_row[side]), int(ring_row[(side + 1) % 3]))
             for owner, ring_row in zip(cell, ring) for side in range(3)}
    return sum(1 for owner, start, end in sides if (owner, end, start) not in sides)


def _check(engine):
    sizes = cell_sizes(engine.corners, engine.faces, engine.face_cells, len(engine.cell_kind))
    assert (sizes > 0).all()
    assert np.isclose(sizes.sum(), _hull(engine), rtol=1e-11)
    assert _open_edges(engine) == 0


def test_the_cells_fill_space_with_no_gaps_or_overlaps_through_collapses_and_splits():
    """Pull at three times its default for collapses, then a split size just below the biggest volumes."""
    engine = new_engine(200, 3, 0.5)
    _check(engine)
    for index in range(30):
        biggest = float(np.sort(engine.grid.volume)[-3:].mean()) / engine.start_mean_volume
        engine = step(engine, EngineSettings.all_on(split_size=biggest if index >= 27 else np.inf,
                                                    attraction=0.015))
        _check(engine)
    assert engine.collapses > 0 and engine.splits > 0
