"""A scene's recording: its frames and why it stopped, if it stopped early.

Contract: `Recording` is a frozen value. `frames[0]` is the starting network,
then one frame per completed step. `stopped` is None when every asked step ran,
else the step that failed and the rules' own message (a rules error or a
non-finite length), shown on the page as it is.
"""
from __future__ import annotations

from dataclasses import dataclass

from physics.viewer.frame import Frame


@dataclass(frozen=True)
class Recording:
    """Frames of one scene and the reason it stopped early, if any."""

    frames: tuple[Frame, ...]
    stopped: str | None
