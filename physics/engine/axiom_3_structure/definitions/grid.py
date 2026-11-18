"""Grid: which cells lie next to which, from their place alone. No table is kept.

A cell's number is its place: `grid` places along three directions, the last counted
first. Energy volumes stand at the places that are even in all three, and a joint at each
place that is odd in one: it lies along that direction, between the two volumes beside it.
The other places hold no cell: there the joints meet. The grid wraps round in all three
directions, so the place after the last is the first and nothing lies at an edge. A row
with one joint left empty is a line of cells.
"""
import warp as wp


@wp.func
def place_of(grid: wp.vec3i, cell: wp.int32, axis: wp.int32) -> wp.int32:
    """The cell's place along `axis` (0, 1 or 2)."""
    if axis == 0:
        return cell // (grid[1] * grid[2])
    if axis == 1:
        return cell // grid[2] % grid[1]
    return cell % grid[2]


@wp.func
def cell_beside(grid: wp.vec3i, cell: wp.int32, axis: wp.int32, way: wp.int32) -> wp.int32:
    """The cell one place from `cell` along `axis` (`way` -1 or 1), round the grid."""
    apart = wp.vec3i(grid[1] * grid[2], grid[2], 1)[axis]
    place = place_of(grid, cell, axis)
    return cell + ((place + way + grid[axis]) % grid[axis] - place) * apart


@wp.func
def axis_of(grid: wp.vec3i, cell: wp.int32) -> wp.int32:
    """The direction a joint's place lies along, or -1 where no joint can stand."""
    odd = wp.vec3i(place_of(grid, cell, 0) % 2, place_of(grid, cell, 1) % 2,
                    place_of(grid, cell, 2) % 2)
    if odd[0] + odd[1] + odd[2] != 1:
        return -1
    return odd[1] + 2 * odd[2]


@wp.func
def volume_beside(cell_quanta: wp.array(dtype=wp.int32), grid: wp.vec3i, cell: wp.int32,
                  axis: wp.int32, way: wp.int32) -> wp.int32:
    """The energy volume one place from `cell`, or -1 when none stands there."""
    there = cell_beside(grid, cell, axis, way)
    if cell_quanta[there] <= 0:
        return -1
    return there


def joints_of(grid: tuple, volume: int) -> list[int]:
    """The six places of a volume's joints: one place on along each direction, each way."""
    return [cell_beside(wp.vec3i(*grid), volume, axis, way) for axis in range(3) for way in (-1, 1)]
