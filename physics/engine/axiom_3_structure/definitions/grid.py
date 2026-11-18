"""Grid: which cells lie next to which, from their place alone. No table is kept.

A cell's number is its place in a small fixed piece of the grid: `grid` places along three
directions, the last counted first. Energy volumes stand at the even places and a joint
at the odd place between two of them, so the cell one place on from a volume is its joint
on that side and the cell after it the next volume. The place where four joints meet
holds no cell. Only the first two directions are built: the piece is a flat cut.
"""
import warp as wp


@wp.func
def cell_beside(grid: wp.vec3i, cell: wp.int32, axis: wp.int32, way: wp.int32) -> wp.int32:
    """The cell one place from `cell` along `axis` (0 or 1; `way` -1 or 1), -1 past the edge."""
    apart = grid[2]
    place = cell // grid[2] % grid[1]
    if axis == 0:
        apart = grid[1] * grid[2]
        place = cell // apart
    if place + way < 0 or place + way >= grid[axis]:
        return -1
    return cell + way * apart


@wp.func
def volume_beside(cell_quanta: wp.array(dtype=wp.int32), grid: wp.vec3i, cell: wp.int32,
                  axis: wp.int32, way: wp.int32) -> wp.int32:
    """The energy volume one place from `cell`, or -1 when none stands there."""
    there = cell_beside(grid, cell, axis, way)
    if there >= 0 and cell_quanta[there] <= 0:
        return -1
    return there


def joints_of(grid: tuple, volume: int) -> list[int]:
    """A volume's joints inside the piece: the cell on each of its sides in the flat cut."""
    found = [cell_beside(wp.vec3i(*grid), volume, axis, way) for axis in (0, 1) for way in (-1, 1)]
    return [cell for cell in found if cell >= 0]
