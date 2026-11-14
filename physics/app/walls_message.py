"""The "walls" message: the 3D glass foam, each joint drawn as a layer of its own. Display only.

Contract: `walls_message(positions, radius, joints, joint_volume, squeeze, step)`
takes a frame's (n, 3) display positions and radii, the (k, 2) joint list and
the frame's per-joint "joint_volume" (share of the two energy volumes' mean
volume) and "squeeze" (0..1). The energy volumes' 3D power diagram
(`power_faces`, weights radius^2, box = the positions' extent widened by the
largest radius) gives the cells and the faces they share. Only faces between
two energy volumes joined by a joint are sent; the share of all power-diagram
faces that are no joint is "mismatch" (0 when there is no face). Returns
{"type": "walls", "step", "faces": [{"a", "b", "polygon": [x, y, z, ...],
"thickness", "squeeze"}], "mismatch"}. A face is the joint layer's middle
plane: the page offsets it by thickness / 2 to both sides along b - a.
thickness = the joint's display volume (share times the two energy volumes'
mean display volume, 4/3 pi r^3) over the face area, clamped to
0..MAX_THICKNESS_SHARE times the smaller radius of a and b.
Coordinates have 3 decimals, thickness and mismatch 4.
"""
from __future__ import annotations

import numpy as np

from physics.app.joint_index import joint_index
from physics.app.power_faces import power_faces

MAX_THICKNESS_SHARE = 0.5


def walls_message(positions: np.ndarray, radius, joints: np.ndarray, joint_volume,
                  squeeze, step: int) -> dict:
    """The joint faces of the 3D power diagram of the frame, as joint layers."""
    message = {"type": "walls", "step": int(step), "faces": [], "mismatch": 0.0}
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    radius = np.asarray(radius, dtype=float)
    joints = np.asarray(joints, dtype=np.int64).reshape(-1, 2)
    if len(positions) < 2:
        return message
    reach = float(radius.max())
    box = (*(positions.min(axis=0) - reach), *(positions.max(axis=0) + reach))
    pairs, polygons = power_faces(positions, radius ** 2, box, joints)
    if not len(pairs):
        return message
    joint_of = np.asarray(joint_index(joints, len(positions), pairs))
    message["mismatch"] = round(float(np.mean(joint_of < 0)), 4)
    drawn = [index for index, polygon in enumerate(polygons)
             if polygon is not None and joint_of[index] >= 0]
    if not drawn:
        return message
    rings = [polygons[index] for index in drawn]
    first, second = pairs[drawn, 0], pairs[drawn, 1]
    joint = joint_of[drawn]
    shares = np.asarray(joint_volume, dtype=float)[joint]
    mean_volume = (4.0 / 3.0) * np.pi * (radius[first] ** 3 + radius[second] ** 3) / 2
    thickness = shares * mean_volume / np.maximum(_areas(rings), 1e-12)
    ceiling = MAX_THICKNESS_SHARE * np.minimum(radius[first], radius[second])
    thickness = np.round(np.clip(np.nan_to_num(thickness), 0.0, ceiling), 4).tolist()
    flat = np.round(np.concatenate(rings), 3).ravel().tolist()
    ends = 3 * np.cumsum([len(ring) for ring in rings])
    starts = np.concatenate([[0], ends[:-1]])
    message["faces"] = [{"a": int(first[index]), "b": int(second[index]),
                         "polygon": flat[starts[index]:ends[index]],
                         "thickness": thickness[index], "squeeze": squeeze[joint[index]]}
                        for index in range(len(drawn))]
    return message


def _areas(rings):
    """Area of each planar polygon, from the cross products of spokes out of its corner mean."""
    sizes = np.array([len(ring) for ring in rings])
    group = np.repeat(np.arange(len(rings)), sizes)
    corners = np.concatenate(rings)
    centre = np.stack([np.bincount(group, corners[:, axis]) for axis in range(3)], axis=1)
    spokes = corners - (centre / sizes[:, None])[group]
    index = np.arange(len(corners))
    following = index + 1
    following[np.cumsum(sizes) - 1] = np.cumsum(sizes) - sizes
    crossed = np.cross(spokes, spokes[following])
    summed = np.stack([np.bincount(group, crossed[:, axis], len(rings)) for axis in range(3)], axis=1)
    return 0.5 * np.linalg.norm(summed, axis=1)
