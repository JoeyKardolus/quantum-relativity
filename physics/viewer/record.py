"""Run a scene and record a frame per step for the viewer.

Contract: `record(scene, steps)` steps `scene.network` with `scene.settings`
up to `steps` times (rng seeded from scene.seed) and returns a Recording with
frame 0 plus one frame per completed step. Display positions start from
`initial_layout` relaxed START_ITERATIONS times, then each step are carried
over collapses (`carry_positions`, energy-weighted, via the stepped network's
merged_into when it has one) and relaxed FRAME_ITERATIONS times toward the new
lengths. The far background (`background_joints`, FAR mean lengths from every
marked joint at the start) is carried over collapses too and gives each
frame's background_length. Positions live only in the frames, never on a network. A step that
raises ValueError or leaves a length that is not finite and positive ends the
recording; its message goes in `stopped`.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from physics.measure import mean_length
from physics.network import Network
from physics.rules import step
from physics.viewer.background import background_joints
from physics.viewer.carry_positions import carry_positions
from physics.viewer.frame import Frame
from physics.viewer.initial_layout import initial_layout
from physics.viewer.joint_energy import joint_energy
from physics.viewer.recording import Recording
from physics.viewer.relax_layout import relax_layout
from physics.viewer.scene import Scene

START_ITERATIONS = 60
FRAME_ITERATIONS = 6
BRIGHT_CONNECTIONS = 800
FAR = 4.0               # far background: mean lengths from every marked joint at the start


def record(scene: Scene, steps: int) -> Recording:
    """Step the scene, laying out and measuring every frame; stop on a rules failure."""
    rng = np.random.default_rng(scene.seed)
    network, marked = scene.network, scene.marked
    far = background_joints(network, marked, FAR)
    positions = relax_layout(network, initial_layout(network), rng, START_ITERATIONS)
    frames = [_frame(0, network, positions, marked, far)]
    for number in range(1, steps + 1):
        try:
            with np.errstate(all="ignore"):
                stepped = step(network, scene.settings, rng)
        except ValueError as error:
            return Recording(tuple(frames), f"step {number}: {error}")
        if not (np.all(np.isfinite(stepped.length)) and np.all(stepped.length > 0)):
            return Recording(tuple(frames), f"step {number}: a connection length stopped being finite and positive")
        merged_into = getattr(stepped, "merged_into", None)
        positions = carry_positions(positions, joint_energy(network), merged_into, stepped.joint_count)
        if merged_into is not None:
            marked = tuple(int(merged_into[joint]) for joint in marked)
            far = np.unique(merged_into[far])
        network = stepped
        positions = relax_layout(network, positions, rng, FRAME_ITERATIONS)
        frames.append(_frame(number, network, positions, marked, far))
    return Recording(tuple(frames), None)


def _frame(number: int, network: Network, positions: np.ndarray, marked: tuple[int, ...],
           far: np.ndarray) -> Frame:
    brightest = np.argsort(network.energy)[::-1][:BRIGHT_CONNECTIONS]
    return Frame(number, network.joint_count, positions - positions.mean(axis=0),
                 joint_energy(network), network.pairs[brightest], marked,
                 _gap(network, marked), float(mean_length(network)), _background_length(network, far))


def _background_length(network: Network, far: np.ndarray) -> float:
    inside = np.zeros(network.joint_count, dtype=bool)
    inside[far] = True
    held = inside[network.pairs[:, 0]] & inside[network.pairs[:, 1]]
    return float(network.length[held].mean()) if held.any() else float("nan")


def _gap(network: Network, marked: tuple[int, ...]) -> float | None:
    if len(marked) < 2:
        return None
    return float(dijkstra(network.adjacency(), directed=False, indices=marked[0])[marked[1]])
