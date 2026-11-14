"""Command line: `python -m physics.viewer [--scene two-lumps|collapse|all] [--out DIR]`.

Contract: records each chosen scene for STEPS steps, writes DIR/index.html
(default DIR /home/dupe/quantum-relativity-runs/viewer) and prints one line per
scene (what happened) plus the page path and its size.
"""
from __future__ import annotations

import argparse
import os

from physics.viewer import build_page, collapse, record, two_lumps, what_happened

STEPS = 100
_SCENES = {"two-lumps": two_lumps, "collapse": collapse}


def main() -> None:
    """Record the chosen scenes and write the viewer page."""
    parser = argparse.ArgumentParser(prog="python -m physics.viewer")
    parser.add_argument("--scene", choices=[*_SCENES, "all"], default="all")
    parser.add_argument("--out", default="/home/dupe/quantum-relativity-runs/viewer")
    arguments = parser.parse_args()
    names = list(_SCENES) if arguments.scene == "all" else [arguments.scene]
    entries = []
    for name in names:
        scene = _SCENES[name]()
        recording = record(scene, STEPS)
        happened = what_happened(recording)
        print(f"{name}: {happened}")
        entries.append((scene, recording, happened))
    path = build_page(entries, os.path.join(arguments.out, "index.html"))
    print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
