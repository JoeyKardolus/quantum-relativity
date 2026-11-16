"""The ladder's Warp kernels: coarse encode and decode, reading any level, copying rows, quietness.

Box order inside a COARSE row is group-major: group g = (x//4)*4 + (y//4)*2 + z//4, place
j = ((x%4)*4 + y%4)*4 + z%4 in the group, k = 64 g + j. Sign codes (0 zero, 1 plus,
2 minus) pack four to a byte at bits 2(k%4) of byte k//4; magnitude and phase offsets
pack two to a byte at bits 4(k%2) of byte k//2. Head per group and sign (side 0 plus,
1 minus): lo, step, rest at 6g + 3 side. Phase head per group: centre, span at 2g.

Exact totals: a box of magnitude m decodes to lo + floor((m - lo)/step) step, never
more, so the group's rest = sum of what floor dropped is >= 0 and is stored exactly;
the rest goes back one quantum at a time in rank order over that sign's boxes, every
box rest // count, the first rest % count one more. Sum of decoded magnitudes equals
the original exactly, per group and per sign, and each box is off by at most step - 1.
"""
from __future__ import annotations

import warp as wp

_TWO_PI = wp.constant(6.283185307179586)


@wp.func
def _wrap(angle: float) -> float:
    """Angle wrapped into [-pi, pi]."""
    return angle - _TWO_PI * wp.round(angle / _TWO_PI)


@wp.func
def _sign_code(sign: wp.array2d(dtype=wp.uint8), row: int, place: int) -> int:
    """The 2-bit sign code of box `place` (group-major) in a COARSE row."""
    return (wp.int32(sign[row, place // 4]) >> (2 * (place % 4))) & 3


@wp.func
def _group_place(ax: int, ay: int, az: int) -> int:
    """Group-major place of box (ax, ay, az) in its block."""
    group = (ax // 4) * 4 + (ay // 4) * 2 + az // 4
    return group * 64 + ((ax % 4) * 4 + ay % 4) * 4 + az % 4


@wp.func
def coarse_quanta(sign: wp.array2d(dtype=wp.uint8), mag: wp.array2d(dtype=wp.uint8),
                  head: wp.array2d(dtype=wp.int32), row: int, ax: int, ay: int, az: int) -> int:
    """Decoded whole quanta of box (ax, ay, az) of COARSE row `row`, rest included."""
    place = _group_place(ax, ay, az)
    group = place // 64
    code = _sign_code(sign, row, place)
    result = int(0)
    if code != 0:
        base = group * 6 + (code - 1) * 3
        low = head[row, base]
        step = head[row, base + 1]
        rest = head[row, base + 2]
        offset = (wp.int32(mag[row, place // 2]) >> (4 * (place % 2))) & 15
        count = int(0)
        rank = int(0)
        for other in range(64):
            if _sign_code(sign, row, group * 64 + other) == code:
                count += 1
                if other < place - group * 64:
                    rank += 1
        share = rest // count
        if rank < rest - share * count:
            share += 1
        result = low + offset * step + share
        if code == 2:
            result = -result
    return result


@wp.func
def coarse_phase(phase: wp.array2d(dtype=wp.uint8), phead: wp.array2d(dtype=wp.float32),
                 row: int, ax: int, ay: int, az: int) -> float:
    """Decoded phase of box (ax, ay, az) of COARSE row `row`, wrapped into [-pi, pi]."""
    place = _group_place(ax, ay, az)
    group = place // 64
    offset = (wp.int32(phase[row, place // 2]) >> (4 * (place % 2))) & 15
    return _wrap(phead[row, group * 2] + (float(offset) / 7.5 - 1.0) * phead[row, group * 2 + 1])


@wp.kernel
def gather_kernel(level: wp.array3d(dtype=wp.int32), slot: wp.array3d(dtype=wp.int32),
                  full_q: wp.array4d(dtype=wp.int32), full_p: wp.array4d(dtype=wp.float32),
                  shared_q: wp.array4d(dtype=wp.int32), shared_p: wp.array4d(dtype=wp.float32),
                  c_sign: wp.array2d(dtype=wp.uint8), c_mag: wp.array2d(dtype=wp.uint8),
                  c_phase: wp.array2d(dtype=wp.uint8), c_head: wp.array2d(dtype=wp.int32),
                  c_phead: wp.array2d(dtype=wp.float32), wanted: wp.array2d(dtype=wp.int32), halo: int,
                  out_q: wp.array4d(dtype=wp.int32), out_p: wp.array4d(dtype=wp.float32)):
    """Every box of each wanted block plus `halo` boxes round it, read from whatever level holds it."""
    item, ax, ay, az = wp.tid()
    gx = wanted[item, 0] * 8 - halo + ax
    gy = wanted[item, 1] * 8 - halo + ay
    gz = wanted[item, 2] * 8 - halo + az
    quanta = int(0)
    angle = float(0.0)
    inside = gx >= 0 and gy >= 0 and gz >= 0
    inside = inside and gx < level.shape[0] * 8 and gy < level.shape[1] * 8 and gz < level.shape[2] * 8
    if inside:
        bx = gx // 8
        by = gy // 8
        bz = gz // 8
        lx = gx - bx * 8
        ly = gy - by * 8
        lz = gz - bz * 8
        code = level[bx, by, bz]
        row = slot[bx, by, bz]
        if code == 1:
            quanta = full_q[row, lx, ly, lz]
            angle = full_p[row, lx, ly, lz]
        if code == 2:
            quanta = coarse_quanta(c_sign, c_mag, c_head, row, lx, ly, lz)
            angle = coarse_phase(c_phase, c_phead, row, lx, ly, lz)
        if code == 3:
            quanta = shared_q[row, lx, ly, lz]
            angle = shared_p[row, lx, ly, lz]
    out_q[item, ax, ay, az] = quanta
    out_p[item, ax, ay, az] = angle


@wp.kernel
def copy_rows_kernel(src_q: wp.array4d(dtype=wp.int32), src_p: wp.array4d(dtype=wp.float32),
                     src_rows: wp.array(dtype=wp.int32), dst_q: wp.array4d(dtype=wp.int32),
                     dst_p: wp.array4d(dtype=wp.float32), dst_rows: wp.array(dtype=wp.int32)):
    """Copy whole 8^3 rows, row src_rows[n] to row dst_rows[n]."""
    item, ax, ay, az = wp.tid()
    dst_q[dst_rows[item], ax, ay, az] = src_q[src_rows[item], ax, ay, az]
    dst_p[dst_rows[item], ax, ay, az] = src_p[src_rows[item], ax, ay, az]


@wp.kernel
def decode_kernel(c_sign: wp.array2d(dtype=wp.uint8), c_mag: wp.array2d(dtype=wp.uint8),
                  c_phase: wp.array2d(dtype=wp.uint8), c_head: wp.array2d(dtype=wp.int32),
                  c_phead: wp.array2d(dtype=wp.float32), src_rows: wp.array(dtype=wp.int32),
                  dst_q: wp.array4d(dtype=wp.int32), dst_p: wp.array4d(dtype=wp.float32),
                  dst_rows: wp.array(dtype=wp.int32)):
    """Decode COARSE rows into FULL rows."""
    item, ax, ay, az = wp.tid()
    row = src_rows[item]
    dst_q[dst_rows[item], ax, ay, az] = coarse_quanta(c_sign, c_mag, c_head, row, ax, ay, az)
    dst_p[dst_rows[item], ax, ay, az] = coarse_phase(c_phase, c_phead, row, ax, ay, az)


@wp.func
def _box_of(group: int, place: int) -> wp.vec3i:
    """Block coordinates of place `place` (0..63) of group `group`."""
    return wp.vec3i((group // 4) * 4 + place // 16, ((group // 2) % 2) * 4 + (place // 4) % 4,
                    (group % 2) * 4 + place % 4)


@wp.kernel
def encode_kernel(full_q: wp.array4d(dtype=wp.int32), full_p: wp.array4d(dtype=wp.float32),
                  src_rows: wp.array(dtype=wp.int32), dst_rows: wp.array(dtype=wp.int32),
                  max_rel_error: float, max_phase_error: float,
                  c_sign: wp.array2d(dtype=wp.uint8), c_mag: wp.array2d(dtype=wp.uint8),
                  c_phase: wp.array2d(dtype=wp.uint8), c_head: wp.array2d(dtype=wp.int32),
                  c_phead: wp.array2d(dtype=wp.float32), within: wp.array2d(dtype=wp.int32)):
    """Encode one group of 64 boxes of a FULL row into a COARSE row; within = 1 when its error bound holds."""
    item, group = wp.tid()
    src = src_rows[item]
    dst = dst_rows[item]
    good = int(1)
    for side in range(2):
        low = int(2147483647)
        high = int(0)
        count = int(0)
        for place in range(64):
            box = _box_of(group, place)
            size = full_q[src, box[0], box[1], box[2]]
            if side == 1:
                size = -size
            if size > 0:
                low = wp.min(low, size)
                high = wp.max(high, size)
                count += 1
        step = int(1)
        rest = int(0)
        if count == 0:
            low = 0
        else:
            step = wp.max(1, (high - low + 14) // 15)
            for place in range(64):
                box = _box_of(group, place)
                size = full_q[src, box[0], box[1], box[2]]
                if side == 1:
                    size = -size
                if size > 0:
                    rest += (size - low) - ((size - low) // step) * step
            if float(step - 1) > max_rel_error * float(low):
                good = 0
        c_head[dst, group * 6 + side * 3] = low
        c_head[dst, group * 6 + side * 3 + 1] = step
        c_head[dst, group * 6 + side * 3 + 2] = rest
    across = float(0.0)
    up = float(0.0)
    for place in range(64):
        box = _box_of(group, place)
        if full_q[src, box[0], box[1], box[2]] != 0:
            angle = full_p[src, box[0], box[1], box[2]]
            across += wp.cos(angle)
            up += wp.sin(angle)
    centre = wp.atan2(up, across)
    span = float(0.0)
    for place in range(64):
        box = _box_of(group, place)
        if full_q[src, box[0], box[1], box[2]] != 0:
            span = wp.max(span, wp.abs(_wrap(full_p[src, box[0], box[1], box[2]] - centre)))
    if span / 15.0 > max_phase_error:
        good = 0
    c_phead[dst, group * 2] = centre
    c_phead[dst, group * 2 + 1] = span
    for byte in range(16):
        packed = int(0)
        for part in range(4):
            box = _box_of(group, byte * 4 + part)
            size = full_q[src, box[0], box[1], box[2]]
            code = int(0)
            if size > 0:
                code = 1
            if size < 0:
                code = 2
            packed = packed | (code << (2 * part))
        c_sign[dst, group * 16 + byte] = wp.uint8(packed)
    for byte in range(32):
        packed_mag = int(0)
        packed_phase = int(0)
        for part in range(2):
            box = _box_of(group, byte * 2 + part)
            size = full_q[src, box[0], box[1], box[2]]
            offset = int(0)
            turn = int(7)
            if size != 0:
                which = int(0)
                if size < 0:
                    which = 1
                    size = -size
                low_side = c_head[dst, group * 6 + which * 3]
                step_side = c_head[dst, group * 6 + which * 3 + 1]
                offset = wp.min(15, (size - low_side) // step_side)
                if span > 0.0:
                    shift = _wrap(full_p[src, box[0], box[1], box[2]] - centre)
                    turn = wp.clamp(int(wp.round((shift / span + 1.0) * 7.5)), 0, 15)
            packed_mag = packed_mag | (offset << (4 * part))
            packed_phase = packed_phase | (turn << (4 * part))
        c_mag[dst, group * 32 + byte] = wp.uint8(packed_mag)
        c_phase[dst, group * 32 + byte] = wp.uint8(packed_phase)
    within[item, group] = good


@wp.func
def _face_step(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32), item: int,
               ax: int, ay: int, az: int, di: int, dj: int, dk: int) -> float:
    """|wrapped phase step| from box (ax, ay, az) to its neighbour; 0 when the neighbour holds no space."""
    step = float(0.0)
    if quanta[item, ax + di, ay + dj, az + dk] != 0:
        step = wp.abs(_wrap(phase[item, ax + di, ay + dj, az + dk] - phase[item, ax, ay, az]))
    return step


@wp.func
def _square_turns(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32), item: int,
                  ax: int, ay: int, az: int, ai: int, aj: int, ak: int, bi: int, bj: int, bk: int) -> int:
    """1 when the unit square at corner (ax, ay, az) spanned by a and b holds a whole turn of phase, else 0."""
    edge = quanta.shape[1]
    hit = int(0)
    fits = ax >= 0 and ay >= 0 and az >= 0 and ax + ai + bi < edge and ay + aj + bj < edge and az + ak + bk < edge
    if fits:
        q0 = quanta[item, ax, ay, az]
        q1 = quanta[item, ax + ai, ay + aj, az + ak]
        q2 = quanta[item, ax + ai + bi, ay + aj + bj, az + ak + bk]
        q3 = quanta[item, ax + bi, ay + bj, az + bk]
        if q0 != 0 and q1 != 0 and q2 != 0 and q3 != 0:
            p0 = phase[item, ax, ay, az]
            p1 = phase[item, ax + ai, ay + aj, az + ak]
            p2 = phase[item, ax + ai + bi, ay + aj + bj, az + ak + bk]
            p3 = phase[item, ax + bi, ay + bj, az + bk]
            total = _wrap(p1 - p0) + _wrap(p2 - p1) + _wrap(p3 - p2) + _wrap(p0 - p3)
            if wp.abs(total) > 3.14159265:
                hit = 1
    return hit


@wp.kernel
def quiet_kernel(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32),
                 flow: wp.array(dtype=wp.float32), whirls: wp.array(dtype=wp.int32),
                 filled: wp.array(dtype=wp.int32)):
    """Per block (gathered with halo 1): boxes with space, largest face phase step, whole-turn squares touching it."""
    item, tx, ty, tz = wp.tid()
    ax = tx + 1
    ay = ty + 1
    az = tz + 1
    if quanta[item, ax, ay, az] != 0:
        wp.atomic_add(filled, item, 1)
        biggest = _face_step(quanta, phase, item, ax, ay, az, 1, 0, 0)
        biggest = wp.max(biggest, _face_step(quanta, phase, item, ax, ay, az, -1, 0, 0))
        biggest = wp.max(biggest, _face_step(quanta, phase, item, ax, ay, az, 0, 1, 0))
        biggest = wp.max(biggest, _face_step(quanta, phase, item, ax, ay, az, 0, -1, 0))
        biggest = wp.max(biggest, _face_step(quanta, phase, item, ax, ay, az, 0, 0, 1))
        biggest = wp.max(biggest, _face_step(quanta, phase, item, ax, ay, az, 0, 0, -1))
        wp.atomic_max(flow, item, biggest)
        turns = int(0)
        for corner in range(4):
            back_a = corner % 2
            back_b = corner // 2
            turns += _square_turns(quanta, phase, item, ax - back_a, ay - back_b, az, 1, 0, 0, 0, 1, 0)
            turns += _square_turns(quanta, phase, item, ax, ay - back_a, az - back_b, 0, 1, 0, 0, 0, 1)
            turns += _square_turns(quanta, phase, item, ax - back_b, ay, az - back_a, 0, 0, 1, 1, 0, 0)
        if turns > 0:
            wp.atomic_add(whirls, item, turns)


@wp.kernel
def total_kernel(quanta: wp.array4d(dtype=wp.int32), totals: wp.array(dtype=wp.int64)):
    """Add every box's magnitude to totals[0] (plus) or totals[1] (minus)."""
    item, ax, ay, az = wp.tid()
    value = quanta[item, ax, ay, az]
    if value > 0:
        wp.atomic_add(totals, 0, wp.int64(value))
    if value < 0:
        wp.atomic_add(totals, 1, wp.int64(-value))


@wp.kernel
def differs_kernel(old_q: wp.array4d(dtype=wp.int32), old_p: wp.array4d(dtype=wp.float32),
                   new_q: wp.array4d(dtype=wp.int32), new_p: wp.array4d(dtype=wp.float32),
                   changed: wp.array(dtype=wp.int32)):
    """changed[n] = 1 when any box of block n differs in quanta or phase bits."""
    item, ax, ay, az = wp.tid()
    if old_q[item, ax, ay, az] != new_q[item, ax, ay, az] or old_p[item, ax, ay, az] != new_p[item, ax, ay, az]:
        changed[item] = 1
