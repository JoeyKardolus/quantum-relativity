"""One PNG of the network before and after a run, side by side.

Contract: `network_picture(path, title, panels, carried)` draws each panel
(caption, network, joints) as its connections among `joints`, coloured by
connection energy on one shared log scale, at positions from `display_layout`.
`carried[i]` is where the first panel's joints[i] ended up in each later panel's
network (one array per later panel); later panels are rotated or mirrored
(never scaled) to best match the first. Each panel is framed to its own extent,
and its caption gains that extent as a multiple of the first panel's, so a
network that shrank a thousandfold is still drawn legibly and the shrink is
stated. The colour scale is shared. Writes the PNG to `path`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from physics.experiments.picture.layout import display_layout  # noqa: E402


def network_picture(path: str, title: str, panels: list[tuple], carried: list[np.ndarray]) -> None:
    """Draw the panels side by side, aligned to the first, and save the PNG."""
    first_joints = panels[0][2]
    first = display_layout(panels[0][1], first_joints)
    drawn = [(panels[0][1], first_joints, first - first.mean(axis=0))]
    for (_, network, joints), mapping in zip(panels[1:], carried):
        positions = display_layout(network, joints)
        drawn.append((network, joints, _aligned(positions, joints, mapping, drawn[0][2])))
    segments = [_segments(network, joints, positions) for network, joints, positions in drawn]
    energies = np.concatenate([energy for _, energy in segments])
    norm = LogNorm(np.percentile(energies, 1), np.percentile(energies, 99.5))
    reaches = [max(float(np.abs(positions).max()), 1e-300) * 1.05 for _, _, positions in drawn]
    figure, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6), dpi=110)
    for axis, (caption, _, _), (lines, energy), reach in zip(axes, panels, segments, reaches):
        order = np.argsort(energy)
        axis.add_collection(LineCollection(lines[order], array=energy[order], cmap="inferno",
                                           norm=norm, linewidths=0.5))
        axis.set_xlim(-reach, reach)
        axis.set_ylim(-reach, reach)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{caption}\nwidth {reach / reaches[0]:.3g} x start", fontsize=10)
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"), ax=list(axes),
                    shrink=0.8, label="connection energy (log)")
    figure.suptitle(title, fontsize=11)
    figure.savefig(path)
    plt.close(figure)


def _aligned(positions: np.ndarray, joints: np.ndarray, mapping: np.ndarray,
             reference: np.ndarray) -> np.ndarray:
    """Rotate/mirror positions so carried joints sit as near their first-panel places as possible."""
    slot = {int(joint): index for index, joint in enumerate(joints)}
    pairs = [(index, slot[int(joint)]) for index, joint in enumerate(mapping) if int(joint) in slot]
    before = reference[[index for index, _ in pairs]]
    after = positions[[index for _, index in pairs]]
    centre = after.mean(axis=0)
    left, _, right = np.linalg.svd((after - centre).T @ (before - before.mean(axis=0)))
    return (positions - centre) @ (left @ right)


def _segments(network, joints: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slot = np.full(network.joint_count, -1)
    slot[joints] = np.arange(len(joints))
    ends = slot[network.pairs]
    inside = (ends >= 0).all(axis=1)
    return positions[ends[inside]], network.energy[inside]
