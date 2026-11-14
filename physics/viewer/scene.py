"""A scene: a starting network, the rules it runs under, and what to watch for.

Contract: `Scene` is a frozen value. `network` is the starting Network,
`settings` the RuleSettings stepped each frame, `marked` the joints
the scene follows (lump centres), `watch_for` one plain sentence for the page,
`seed` the seed for its build and steps.
"""
from __future__ import annotations

from dataclasses import dataclass

from physics.network import Network
from physics.rules import RuleSettings


@dataclass(frozen=True)
class Scene:
    """Starting network, rules, followed joints and the sentence saying what to watch for."""

    name: str
    title: str
    network: Network
    settings: RuleSettings
    marked: tuple[int, ...]
    watch_for: str
    seed: int
