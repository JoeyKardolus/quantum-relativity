"""Grid: every array as its spec says, one block per id for good, relations and ways as tables."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.neighbour_of import neighbour_code, neighbour_count, neighbour_id
from physics.engine.axiom_3_structure.definitions.structure_arrays import ARRAYS, HOST
from physics.engine.axiom_3_structure.definitions.tests.conftest import CODES, lattice_tables, start_grid_by_hand
from physics.engine.axiom_3_structure.logic.relink import relink

EXTRA = {"quanta": ("int32", "box", False), "way_turn": ("float32", "way", False),
         "found": ("int32", 2, True), "cell_quanta": ("int64", "cell", False)}


@wp.kernel
def _read_entries(adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32), adj_id: wp.array(dtype=wp.int32),
                  adj_code: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
                  ids: wp.array2d(dtype=wp.int32), codes: wp.array2d(dtype=wp.int32)):
    cell = wp.tid()
    counts[cell] = neighbour_count(adj_n, cell)
    for index in range(counts[cell]):
        ids[cell, index] = neighbour_id(adj_at, adj_id, cell, index)
        codes[cell, index] = neighbour_code(adj_at, adj_code, cell, index)


def _device_ways(grid: Grid) -> set:
    """The living ways as the device arrays hold them."""
    columns = [grid.host(name) for name in ("way_a", "way_b", "way_mouth_a", "way_mouth_b", "way_joint", "way_pass")]
    return {tuple(int(column[slot]) for column in columns) + (slot,)
            for slot in range(grid.way_count) if columns[4][slot] >= 0}


def _shared_mouths(grid: Grid) -> int:
    """Pairs of living ways with one pass on one mouth of one volume."""
    seen, shared = set(), 0
    for way_a, way_b, mouth_a, mouth_b, _, way_pass, _ in _device_ways(grid):
        for key in ((way_a, mouth_a, way_pass), (way_b, mouth_b, way_pass)):
            shared += key in seen
            seen.add(key)
    return shared


def test_every_array_has_its_shape_and_type():
    grid = start_grid_by_hand(extra_arrays=EXTRA)
    assert grid.quanta.shape == (grid.capacity, 6, 6, 6) and grid.quanta.dtype == wp.int32
    assert grid.cell_quanta.shape == (grid.capacity,) and grid.cell_quanta.dtype == wp.int64
    assert grid.way_turn.shape == (grid.way_capacity,) and grid.way_turn.dtype == wp.float32
    assert grid.found.shape == (2,)
    for name, (dtype, shape, each_step) in ARRAYS.items():
        assert getattr(grid, name).dtype == getattr(wp, dtype) and each_step is False
        assert getattr(grid, name).shape == ((grid.capacity,) if shape == "cell" else (grid.way_capacity,))
    assert (grid.count, grid.step, grid.seed, grid.side) == (64, 0, 1, 4)


def test_host_values_are_copies_and_per_cell_arrays_start_empty(grid):
    assert grid.version == 0 and grid.way_passes == 1 and grid.changed == []
    assert grid.changed is not HOST["changed"]
    assert grid.fell_into.dtype == np.int32 and (grid.fell_into == -1).all() and (grid.torn_from == -1).all()
    assert len(grid.made_kind) == grid.capacity and (grid.made_kind[64:] == 0).all()


def test_wrong_inputs_are_refused():
    with pytest.raises(ValueError):
        Grid(5, 200, 10, ARRAYS, HOST, 1, "cpu")
    with pytest.raises(ValueError):
        Grid(4, 10, 10, ARRAYS, HOST, 1, "cpu")
    with pytest.raises(ValueError):
        Grid(4, 64, 10, {"kind": ARRAYS["kind"]}, HOST, 1, "cpu")
    with pytest.raises(ValueError):
        Grid(4, 64, 10, {**ARRAYS, "version": ("int32", "cell", False)}, HOST, 1, "cpu")


def test_the_start_has_26_entries_a_cell_each_with_its_mirror(grid):
    kind = grid.host("kind")
    assert (kind == ENERGY_VOLUME).sum() == 8 and (kind == JOINT).sum() == 56
    for cell in range(grid.count):
        entries = grid.neighbours(cell)
        assert len(entries) == 26 and sorted(code for _, code in entries) == CODES
        for other, code in entries:
            assert (cell, 26 - code) in grid.neighbours(other)
            if kind[cell] == ENERGY_VOLUME:
                assert kind[other] == JOINT                 # plus never touches plus


def test_a_kernel_reads_the_same_entries(grid):
    counts = wp.zeros(grid.count, dtype=wp.int32, device="cpu")
    ids = wp.zeros((grid.count, 26), dtype=wp.int32, device="cpu")
    codes = wp.zeros((grid.count, 26), dtype=wp.int32, device="cpu")
    wp.launch(_read_entries, dim=grid.count,
              inputs=[grid.adj_at, grid.adj_n, grid.adj_id, grid.adj_code, counts, ids, codes], device="cpu")
    assert (counts.numpy() == 26).all()
    for cell in (0, 21, 63):
        assert list(zip(ids.numpy()[cell].tolist(), codes.numpy()[cell].tolist())) == grid.neighbours(cell)


def test_ways_at_the_start_one_through_a_face_two_an_edge_four_a_corner(grid):
    assert grid.way_count == 24 * 1 + 24 * 2 + 8 * 4
    per_type = {1: 1, 2: 2, 3: 4}
    for joint in np.flatnonzero(grid.host("kind") == JOINT):
        odd = sum(part % 2 for part in (joint // 16, (joint // 4) % 4, joint % 4))
        assert len(grid.ways_of(int(joint))) == per_type[odd]
    assert (grid.host("way_pass") == 0).all() and _shared_mouths(grid) == 0
    slot, way_a, way_b, mouth_a, mouth_b = grid.ways_of(16)[0]      # the face joint at (1, 0, 0)
    assert (way_a, way_b) == (0, 32)                                # volumes at (0, 0, 0) and (2, 0, 0)
    assert (mouth_a, mouth_b) == (22, 4)                            # a looks +x at the joint, b looks -x
    assert grid.host("way_joint")[slot] == 16


def test_a_block_belongs_to_one_cell():
    grid = start_grid_by_hand(extra_arrays=EXTRA)
    values = np.arange(216, dtype=np.int32).reshape(6, 6, 6)
    grid.set_block("quanta", 5, values)
    assert np.array_equal(grid.block("quanta", 5), values)
    assert not grid.block("quanta", 4).any() and not grid.block("quanta", 6).any()
    with pytest.raises(ValueError):
        grid.set_block("kind", 5, values)
    with pytest.raises(ValueError):
        grid.block("quanta", grid.count)


def test_growing_keeps_every_row_and_an_id_is_never_used_twice():
    grid = start_grid_by_hand(spare=1, extra_arrays=EXTRA)
    grid.set_block("quanta", 63, np.full((6, 6, 6), 7, dtype=np.int32))
    before_entries, before_ways = grid.neighbours(63), _device_ways(grid)
    first, second = grid.new_ids(3), grid.new_ids(2)
    assert first == [64, 65, 66] and second == [67, 68] and grid.count == 69 and grid.capacity >= 69
    grid.grow(0, 500)
    assert grid.way_capacity >= grid.way_count + 500
    assert (grid.block("quanta", 63) == 7).all() and grid.neighbours(63) == before_entries
    assert _device_ways(grid) == before_ways
    assert (grid.way_joint.numpy()[grid.way_count:] == -1).all()    # an unused slot holds no way
    assert len(grid.fell_into) == grid.capacity and (grid.fell_into == -1).all()
    grid.set_cell("kind", 64, JOINT)
    grid.set_cell("torn_from", 64, 3)
    assert grid.host("kind")[64] == JOINT and grid.host("torn_from")[64] == 3 and grid.neighbours(64) == []


def _fall_into(grid: Grid, joint: int, kept: int, gone: int) -> list:
    """Stage the relations of a collapse by hand: the joint goes and `gone` falls into `kept`."""
    kind = grid.host("kind")
    touched = {joint, kept, gone}
    for other, code in grid.neighbours(joint):
        grid.set_neighbours(other, [entry for entry in grid.neighbours(other) if entry[0] != joint])
        touched.add(other)
    grid.set_neighbours(joint, [])
    mirror = list(grid.neighbours(kept))
    for other, code in grid.neighbours(gone):
        entries = [(kept, there) if one == gone else (one, there) for one, there in grid.neighbours(other)]
        grid.set_neighbours(other, entries)
        mirror.append((other, code))
        touched.add(other)
    grid.set_neighbours(gone, [])
    grid.set_neighbours(kept, mirror)
    grid.set_cell("kind", joint, 0)
    grid.set_cell("kind", gone, 0)
    assert kind[joint] == JOINT
    return sorted(touched)


def test_relink_keeps_living_slots_kills_dead_ways_and_parts_shared_mouths():
    grid = start_grid_by_hand(extra_arrays=EXTRA)
    grid.way_turn.fill_(0.5)
    before = {row[:5]: row[6] for row in _device_ways(grid)}
    touched = _fall_into(grid, joint=16, kept=0, gone=32)
    relink(grid, touched)
    after = _device_ways(grid)
    kind = grid.host("kind")
    assert grid.ways_of(16) == [] and all(row[4] != 16 for row in after)
    assert all(32 not in row[:2] for row in after)                  # no way names the vanished volume
    for row in after:
        if row[:5] in before:
            assert before[row[:5]] == row[6]                        # a way that still exists kept its slot
        else:
            assert grid.host("way_turn")[row[6]] == 0.0             # a new way starts at 0
    assert sum(row[:5] not in before for row in after) > 0
    assert len(grid.neighbours(0)) > 26                             # kept now has several joints at one code
    assert _shared_mouths(grid) == 0 and grid.way_passes == max(row[5] for row in after) + 1 > 1
    wanted = set()
    for joint in np.flatnonzero(kind == JOINT):
        entries = grid.neighbours(int(joint))
        for first, code in entries:
            for second, there in entries:
                if there == 26 - code and first < second and kind[first] == kind[second] == ENERGY_VOLUME:
                    wanted.add((first, second, 26 - code, code, int(joint)))
    assert {row[:5] for row in after} == wanted                     # every way is two opposite entries of its joint
    for cell in touched:                                            # the device holds what the host holds
        first = int(grid.adj_at.numpy()[cell])
        number = int(grid.adj_n.numpy()[cell])
        assert list(zip(grid.adj_id.numpy()[first:first + number].tolist(),
                        grid.adj_code.numpy()[first:first + number].tolist())) == grid.neighbours(cell)


def test_lattice_tables_have_the_pattern():
    kind, ids, ways = lattice_tables(4)
    assert len(ways) == 104 and (kind[ways[:, 4]] == JOINT).all() and (ways[:, 0] < ways[:, 1]).all()
